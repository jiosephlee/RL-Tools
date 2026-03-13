# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Versioned tool registry for GRPO training.

Versions (incremental):
  - v1: RDKit basic + AccFG
  - v2: v1 + remove_salts (standardize_tools)
  - v3: v2 + predict_pka + estimate_logd + get_3d_exposed_polar_surface
  - v4: v2 + predict_pka + estimate_logd + score_structural_alerts (no 3DEPSA)
  - v5: v4 + predict_synthesizability (RAscore) + predict_metabolic_sites (SyGMa)
         + predict_electronic_properties (GFN2-xTB)

Ported from OpenRLHF-Tools/openrlhf/utils/tool_versions.py.

Usage::

    from nemo_rl.environments.tool_calling.tools.tool_versions import get_version

    ver = get_version("v4")
    schemas = ver["basic_schemas"]          # list of OpenAI tool dicts
    task_map = ver["task_specific_map"]     # {task: [extra tool dicts]}
    callables = ver["callables"]            # {name: callable}
"""

import logging
import sys
from typing import Any, Callable, Dict, List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Imports from co-located tool modules (always available — RDKit only)
# ---------------------------------------------------------------------------
from nemo_rl.environments.tool_calling.tools.rdkit_tools import (
    RDKIT_BASIC_OPENAI_TOOLS,
    TDC_RDKIT_SPECIFIC_OPENAI_TOOLS_MAP,
    # basic callables
    get_molecular_weight,
    get_exact_molecular_weight,
    get_heavy_atom_count,
    get_mol_logp,
    get_tpsa,
    get_hbd,
    get_hba,
    get_num_rotatable_bonds,
    get_fraction_csp3,
    get_mol_mr,
    get_ring_count,
    get_num_aromatic_rings,
    get_formal_charge,
    get_qed,
    get_num_heteroatoms,
    # task-specific callables
    get_labute_asa,
    get_max_abs_partial_charge,
    get_min_abs_partial_charge,
    get_max_estate_index,
    get_min_estate_index,
    get_num_aromatic_atoms,
    get_fraction_aromatic_atoms,
    get_num_positive_charge_atoms,
    get_num_negative_charge_atoms,
    get_num_aliphatic_rings,
    get_num_saturated_rings,
    get_num_heterocycles,
    get_num_aromatic_heterocycles,
    get_num_aliphatic_heterocycles,
    get_num_saturated_heterocycles,
    get_num_amide_bonds,
    get_bertz_ct,
    get_balaban_j,
    get_ipc,
    get_hall_kier_alpha,
    get_kappa1,
    get_kappa2,
    get_kappa3,
    get_num_atom_stereo_centers,
    get_num_unspecified_atom_stereo_centers,
)
from nemo_rl.environments.tool_calling.tools.accfg import AccFG_OPENAI_TOOLS, cached_describe_high_level_fg_fragments
from nemo_rl.environments.tool_calling.tools.standardize_tools import STANDARDIZE_OPENAI_TOOLS, remove_salts

# Optional: ePSA (may fail if freesasa not installed)
try:
    from nemo_rl.environments.tool_calling.tools.epsa_3d import get_3d_exposed_polar_surface, SASA_OPENAI_TOOLS
except ImportError:
    get_3d_exposed_polar_surface = None
    SASA_OPENAI_TOOLS = []

# Optional: pKa tools (may fail if molgpka not installed)
try:
    from nemo_rl.environments.tool_calling.tools.pka_related_tools import predict_pka, estimate_logd, PKA_TOOL, LOGD_TOOL
except ImportError:
    predict_pka = None
    estimate_logd = None
    PKA_TOOL = None
    LOGD_TOOL = None

# Optional: Haydn wrappers
try:
    from nemo_rl.environments.tool_calling.tools.haydn_wrappers import HAYDN_OPENAI_TOOLS, HAYDN_CALLABLES
except ImportError:
    HAYDN_OPENAI_TOOLS = []
    HAYDN_CALLABLES = {}

# ---------------------------------------------------------------------------
# Shared callables (RDKit basic + AccFG + task-specific)
# ---------------------------------------------------------------------------
_RDKIT_ACCFG_CALLABLES: Dict[str, Callable] = {
    "describe_high_level_fg_fragments": cached_describe_high_level_fg_fragments,
    "get_molecular_weight": get_molecular_weight,
    "get_exact_molecular_weight": get_exact_molecular_weight,
    "get_heavy_atom_count": get_heavy_atom_count,
    "get_mol_logp": get_mol_logp,
    "get_tpsa": get_tpsa,
    "get_hbd": get_hbd,
    "get_hba": get_hba,
    "get_num_rotatable_bonds": get_num_rotatable_bonds,
    "get_fraction_csp3": get_fraction_csp3,
    "get_labute_asa": get_labute_asa,
    "get_mol_mr": get_mol_mr,
    "get_ring_count": get_ring_count,
    "get_num_aromatic_rings": get_num_aromatic_rings,
    "get_formal_charge": get_formal_charge,
    "get_qed": get_qed,
    "get_num_heteroatoms": get_num_heteroatoms,
    "get_max_abs_partial_charge": get_max_abs_partial_charge,
    "get_min_abs_partial_charge": get_min_abs_partial_charge,
    "get_max_estate_index": get_max_estate_index,
    "get_min_estate_index": get_min_estate_index,
    "get_num_aromatic_atoms": get_num_aromatic_atoms,
    "get_fraction_aromatic_atoms": get_fraction_aromatic_atoms,
    "get_num_positive_charge_atoms": get_num_positive_charge_atoms,
    "get_num_negative_charge_atoms": get_num_negative_charge_atoms,
    "get_num_aliphatic_rings": get_num_aliphatic_rings,
    "get_num_saturated_rings": get_num_saturated_rings,
    "get_num_heterocycles": get_num_heterocycles,
    "get_num_aromatic_heterocycles": get_num_aromatic_heterocycles,
    "get_num_aliphatic_heterocycles": get_num_aliphatic_heterocycles,
    "get_num_saturated_heterocycles": get_num_saturated_heterocycles,
    "get_num_amide_bonds": get_num_amide_bonds,
    "get_bertz_ct": get_bertz_ct,
    "get_balaban_j": get_balaban_j,
    "get_ipc": get_ipc,
    "get_hall_kier_alpha": get_hall_kier_alpha,
    "get_kappa1": get_kappa1,
    "get_kappa2": get_kappa2,
    "get_kappa3": get_kappa3,
    "get_num_atom_stereo_centers": get_num_atom_stereo_centers,
    "get_num_unspecified_atom_stereo_centers": get_num_unspecified_atom_stereo_centers,
}

# ---------------------------------------------------------------------------
# Global exclusion set — tool names listed here are stripped from ALL versions
# (both schemas and callables).
# ---------------------------------------------------------------------------
_EXCLUDED_TOOLS: set = {
    "get_exact_molecular_weight",
}

# Tools that upstream RDKIT_BASIC_OPENAI_TOOLS bundles but we manage via
# Haydn wrappers only.  Strip them from the base import so they don't leak
# into every version; versions that want them add them explicitly from
# HAYDN_OPENAI_TOOLS.
_HAYDN_ONLY_TOOLS: set = {
    "analyze_ring_systems",
    "classify_ionization",
    "compute_similarity",
    "score_structural_alerts",
    "extract_pharmacophore_features",
    "match_substructure",
    "find_mcs",
}

_ALL_STRIPPED: set = _EXCLUDED_TOOLS | _HAYDN_ONLY_TOOLS


def _filter_schemas(schemas: List[Dict[str, Any]], *, haydn_passthrough: bool = False) -> List[Dict[str, Any]]:
    """Remove globally-excluded and (by default) Haydn-only tools from schemas.

    Set *haydn_passthrough=True* when adding Haydn tools explicitly so that
    only the global exclusion set applies.
    """
    blocked = _EXCLUDED_TOOLS if haydn_passthrough else _ALL_STRIPPED
    return [t for t in schemas if t["function"]["name"] not in blocked]


def _filter_callables(callables: Dict[str, Callable], *, haydn_passthrough: bool = False) -> Dict[str, Callable]:
    """Remove globally-excluded and (by default) Haydn-only tools from callables."""
    blocked = _EXCLUDED_TOOLS if haydn_passthrough else _ALL_STRIPPED
    return {k: v for k, v in callables.items() if k not in blocked}


# ---------------------------------------------------------------------------
# Version schemas (incremental)
# ---------------------------------------------------------------------------
_V1_SCHEMAS: List[Dict[str, Any]] = _filter_schemas(RDKIT_BASIC_OPENAI_TOOLS + AccFG_OPENAI_TOOLS)
_V2_SCHEMAS: List[Dict[str, Any]] = _V1_SCHEMAS + _filter_schemas(STANDARDIZE_OPENAI_TOOLS)

_V3_EXTRA_SCHEMAS: List[Dict[str, Any]] = []
if PKA_TOOL is not None:
    _V3_EXTRA_SCHEMAS.append(PKA_TOOL)
if LOGD_TOOL is not None:
    _V3_EXTRA_SCHEMAS.append(LOGD_TOOL)
_V3_SCHEMAS: List[Dict[str, Any]] = _V2_SCHEMAS + _filter_schemas(_V3_EXTRA_SCHEMAS + SASA_OPENAI_TOOLS)

# v4: v2 + pKa + logD + Haydn structural alerts (no 3DEPSA)
_V4_HAYDN_NAMES = {"score_structural_alerts"}

if PKA_TOOL is not None and LOGD_TOOL is not None:
    _V4_EXTRA_SCHEMAS: List[Dict[str, Any]] = [PKA_TOOL, LOGD_TOOL]
    _V4_SCHEMAS: List[Dict[str, Any]] = _V2_SCHEMAS + _filter_schemas(
        _V4_EXTRA_SCHEMAS + [
            t for t in HAYDN_OPENAI_TOOLS if t["function"]["name"] in _V4_HAYDN_NAMES
        ],
        haydn_passthrough=True,
    )
else:
    logger.warning(
        "v4 requires predict_pka and estimate_logd (molgpka). "
        "v4 will fall back to v2 schemas without pKa/logD tools."
    )
    _V4_SCHEMAS = _V2_SCHEMAS

# ---------------------------------------------------------------------------
# Version callables (incremental)
# ---------------------------------------------------------------------------
_V1_CALLABLES: Dict[str, Callable] = _filter_callables(_RDKIT_ACCFG_CALLABLES)

_V2_CALLABLES: Dict[str, Callable] = _filter_callables({**_V1_CALLABLES, "remove_salts": remove_salts})

_V3_CALLABLES: Dict[str, Callable] = dict(_V2_CALLABLES)
if predict_pka is not None:
    _V3_CALLABLES["predict_pka"] = predict_pka
if estimate_logd is not None:
    _V3_CALLABLES["estimate_logd"] = estimate_logd
if get_3d_exposed_polar_surface is not None:
    _V3_CALLABLES["get_3d_exposed_polar_surface"] = get_3d_exposed_polar_surface

_V4_CALLABLES: Dict[str, Callable] = dict(_V2_CALLABLES)
if predict_pka is not None:
    _V4_CALLABLES["predict_pka"] = predict_pka
if estimate_logd is not None:
    _V4_CALLABLES["estimate_logd"] = estimate_logd
_V4_CALLABLES.update({k: v for k, v in HAYDN_CALLABLES.items() if k in _V4_HAYDN_NAMES})

# ---------------------------------------------------------------------------
# Public registry
# ---------------------------------------------------------------------------
TOOL_VERSIONS: Dict[str, Dict[str, Any]] = {
    "v1": {
        "basic_schemas": _V1_SCHEMAS,
        "task_specific_map": TDC_RDKIT_SPECIFIC_OPENAI_TOOLS_MAP,
        "callables": _V1_CALLABLES,
    },
    "v2": {
        "basic_schemas": _V2_SCHEMAS,
        "task_specific_map": TDC_RDKIT_SPECIFIC_OPENAI_TOOLS_MAP,
        "callables": _V2_CALLABLES,
    },
    "v3": {
        "basic_schemas": _V3_SCHEMAS,
        "task_specific_map": TDC_RDKIT_SPECIFIC_OPENAI_TOOLS_MAP,
        "callables": _V3_CALLABLES,
    },
    "v4": {
        "basic_schemas": _V4_SCHEMAS,
        "task_specific_map": TDC_RDKIT_SPECIFIC_OPENAI_TOOLS_MAP,
        "callables": _V4_CALLABLES,
    },
}


def get_version(ver: str) -> Dict[str, Any]:
    """Return the tool version config dict, or raise ValueError."""
    if ver not in TOOL_VERSIONS:
        raise ValueError(
            f"Unknown tool version {ver!r}. "
            f"Available: {sorted(TOOL_VERSIONS.keys())}"
        )
    return TOOL_VERSIONS[ver]


__all__ = ["TOOL_VERSIONS", "get_version"]

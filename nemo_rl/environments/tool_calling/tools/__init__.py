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

"""Tool implementations for TDC molecular property prediction.

Ported from OpenRLHF-Tools/Intern-S1-recipe/tools/ to remove external dependency.
All tools require RDKit; optional dependencies (molgpka, freesasa) are handled
via try/except at import time.
"""

import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)


def get_default_tools() -> dict[str, Callable[..., Any]]:
    """Return dict of all available TDC tool callables.

    Gracefully handles missing dependencies (rdkit, etc.) by returning
    only the tools that can be imported.
    """
    tools: dict[str, Callable[..., Any]] = {}

    try:
        from nemo_rl.environments.tool_calling.tools.rdkit_tools import (
            calculate_logp,
            calculate_molecular_weight,
            calculate_num_hba,
            calculate_num_hbd,
            calculate_num_rotatable_bonds,
            calculate_tpsa,
            get_molecular_formula,
            smiles_to_iupac,
        )

        tools.update({
            "calculate_logp": calculate_logp,
            "calculate_molecular_weight": calculate_molecular_weight,
            "calculate_tpsa": calculate_tpsa,
            "calculate_num_hba": calculate_num_hba,
            "calculate_num_hbd": calculate_num_hbd,
            "calculate_num_rotatable_bonds": calculate_num_rotatable_bonds,
            "get_molecular_formula": get_molecular_formula,
            "smiles_to_iupac": smiles_to_iupac,
        })
    except ImportError as e:
        logger.warning("RDKit tools unavailable: %s", e)

    try:
        from nemo_rl.environments.tool_calling.tools.standardize_tools import (
            standardize_smiles,
        )

        tools["standardize_smiles"] = standardize_smiles
    except ImportError as e:
        logger.warning("Standardize tools unavailable: %s", e)

    if not tools:
        logger.warning("No tool callables available — tool calls will return errors")

    return tools

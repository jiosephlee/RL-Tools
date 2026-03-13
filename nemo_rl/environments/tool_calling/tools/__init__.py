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
            get_mol_logp,
            get_molecular_weight,
            get_hba,
            get_hbd,
            get_num_rotatable_bonds,
            get_tpsa,
        )

        tools.update({
            "get_mol_logp": get_mol_logp,
            "get_molecular_weight": get_molecular_weight,
            "get_tpsa": get_tpsa,
            "get_hba": get_hba,
            "get_hbd": get_hbd,
            "get_num_rotatable_bonds": get_num_rotatable_bonds,
        })
    except ImportError as e:
        logger.warning("RDKit tools unavailable: %s", e)

    try:
        from nemo_rl.environments.tool_calling.tools.standardize_tools import (
            remove_salts,
        )

        tools["remove_salts"] = remove_salts
    except ImportError as e:
        logger.warning("Standardize tools unavailable: %s", e)

    if not tools:
        logger.warning("No tool callables available — tool calls will return errors")

    return tools

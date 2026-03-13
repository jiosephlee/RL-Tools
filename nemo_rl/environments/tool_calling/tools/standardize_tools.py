from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from typing import Dict, Any, List
import json
from .rdkit_tools import _tool

# -------------------------
# Core helper
# -------------------------
def _mol_from_smiles(smiles: str) -> Chem.Mol:
    """
    Parse a SMILES string into an RDKit Mol (sanitized 2D graph).
    Raises ValueError if SMILES is invalid or cannot be sanitized.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("smiles must be a non-empty string")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    return mol


# -------------------------
# Tool function
# -------------------------
def remove_salts(smiles: str) -> str:
    """
    Remove salts/counterions from a SMILES string by keeping the largest
    organic fragment, then return the canonical SMILES of the desalted molecule.

    Uses RDKit's LargestFragmentChooser with preferOrganic=True.
    For single-component SMILES the molecule is returned unchanged.

    Args:
        smiles (str): SMILES string of the molecule (may contain salts, e.g.
                      "CC(=O)O.[Na]" or "[Na+].OC(=O)c1ccccc1").

    Returns:
        str: Canonical SMILES of the largest organic fragment after salt removal.
    """
    mol = _mol_from_smiles(smiles)

    lfc = rdMolStandardize.LargestFragmentChooser(preferOrganic=True)
    cleaned = lfc.choose(mol)
    if cleaned is None:
        raise ValueError(f"Salt removal failed for SMILES: {smiles!r}")

    cleaned_smiles = Chem.MolToSmiles(cleaned, canonical=True, isomericSmiles=True)

    tool_output = f'Canonical SMILES of the largest organic fragment after salt removal: {cleaned_smiles}'

    return tool_output


# -------------------------
# Tool list for OpenAI function-calling
# -------------------------
STANDARDIZE_OPENAI_TOOLS: List[Dict[str, Any]] = [
    _tool(
        "remove_salts",
        (
            "Desalt a SMILES by removing salts/counterions and keeping ONLY the largest organic fragment. "
            "Use this tool when the SMILES contains "
            "multiple disconnected components separated by '.' and one or more components are likely "
            "inorganic ions/solvents (e.g., [Na+], Cl-, Br-, I-, [K+], [Li+], [Ca2+], [Mg2+], [NH4+], "
            "sulfate/phosphate, etc.), or when you want the parent (active) molecule for downstream "
            "property prediction.\n"
            "In aqueous/physiological conditions, salts typically dissociate; "
            "counterions are usually not part of the parent drug structure for many structure-based predictions.\n"
            "Typical examples needing desalting: 'CC(=O)O.[Na+]', '[Na+].O=C(O)c1ccccc1', 'CCO.Cl'.\n"
            "Do NOT use if the SMILES is already a single component (no '.') or if multiple organic "
            "fragments represent real mixture/co-crystal/combination therapy that should be kept. "
            "Output is the canonical SMILES of the retained parent fragment; smaller fragments are discarded."
        )
    ),
]

if __name__ == "__main__":
    print(json.dumps(STANDARDIZE_OPENAI_TOOLS, indent=2))

    smiles = "[Na+].O=C(O)c1ccccc1"
    print(remove_salts(smiles))

    smiles = "CC(=O)O.Cl"
    print(remove_salts(smiles))

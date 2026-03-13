from rdkit import Chem, DataStructs, RDConfig, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors, Lipinski, Crippen, GraphDescriptors, Fragments, QED, rdFingerprintGenerator as rfg, MACCSkeys, ChemicalFeatures
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from typing import Dict, Any, List, cast, TypeVar
from dimorphite_dl import protonate_smiles
from collections import Counter
from enum import StrEnum
from pydantic import BaseModel, Field
try:
    from pydantic_ai import ModelRetry
except ImportError:
    ModelRetry = ValueError
from collections.abc import Callable
import os


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
# OpenAI “tools” schema (ChatCompletions / Responses compatible)
# -------------------------
def _tool(name: str, description: str) -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "smiles": {
                        "type": "string",
                        "description": "SMILES string of the molecule."
                    }
                },
                "required": ["smiles"],
                "additionalProperties": False
            }
        }
    }


# ============================================================
# A) BASIC: widely useful across almost all TDC ADME/Tox tasks
# ============================================================
def get_molecular_weight(smiles: str) -> str:
    """Average molecular weight (Daltons) using RDKit's MolWt."""
    return f"Average molecular weight (Daltons): {float(Descriptors.MolWt(_mol_from_smiles(smiles))):.4f}"

def get_exact_molecular_weight(smiles: str) -> str:
    """Monoisotopic (exact) molecular weight (Daltons) using RDKit's ExactMolWt."""
    return f"Exact molecular weight (Daltons): {float(Descriptors.ExactMolWt(_mol_from_smiles(smiles))):.4f}"

def get_heavy_atom_count(smiles: str) -> str:
    """Number of non-hydrogen atoms (heavy atoms)."""
    return f"Number of heavy atoms: {int(Descriptors.HeavyAtomCount(_mol_from_smiles(smiles)))}"

def get_mol_logp(smiles: str) -> str:
    """
    Wildman–Crippen cLogP estimate (octanol/water partition coefficient).
    Computed from atom fragments + corrections (2D; no 3D needed).
    """
    return f"cLogP estimate: {float(Crippen.MolLogP(_mol_from_smiles(smiles))):.4f}"

def get_tpsa(smiles: str) -> str:
    """
    Topological Polar Surface Area (TPSA, Å^2) using RDKit's fragment-based method.
    Purely 2D/topological; no 3D needed.
    """
    return f"Topological Polar Surface Area (TPSA, Å^2): {float(rdMolDescriptors.CalcTPSA(_mol_from_smiles(smiles))):.4f}"

def get_hbd(smiles: str) -> str:
    """H-bond donors count (Lipinski-style SMARTS rules)."""
    return f"Number of H-bond donors: {int(Lipinski.NumHDonors(_mol_from_smiles(smiles)))}"

def get_hba(smiles: str) -> str:
    """H-bond acceptors count (Lipinski-style SMARTS rules)."""
    return f"Number of H-bond acceptors: {int(Lipinski.NumHAcceptors(_mol_from_smiles(smiles)))}"

def get_num_rotatable_bonds(smiles: str) -> str:
    """
    Rotatable bonds count (Lipinski-style definition; excludes e.g. amide C–N, ring bonds, etc.).
    """
    return f"Number of rotatable bonds: {int(Lipinski.NumRotatableBonds(_mol_from_smiles(smiles)))}"

def get_fraction_csp3(smiles: str) -> str:
    """
    FractionCSP3: (# of sp3 carbon atoms) / (total carbon atoms).
    Often used as a simple “3D character / saturation” proxy from 2D structure.
    """
    return f"Fraction of sp3 carbons: {float(rdMolDescriptors.CalcFractionCSP3(_mol_from_smiles(smiles))):.4f}"

def get_mol_mr(smiles: str) -> str:
    """
    Wildman–Crippen molar refractivity (MR) estimate.
    Proxy for polarizability/volume; fragment-based (2D).
    """
    return f"Wildman–Crippen molar refractivity (MR) estimate: {float(Crippen.MolMR(_mol_from_smiles(smiles))):.4f}"

def get_ring_count(smiles: str) -> str:
    """Total ring count. RDKit: Lipinski.RingCount(mol)."""
    v = int(Lipinski.RingCount(_mol_from_smiles(smiles)))
    return f"Total ring count: {v}"

def get_num_aromatic_rings(smiles: str) -> str:
    """Aromatic ring count. RDKit: Lipinski.NumAromaticRings(mol)."""
    v = int(Lipinski.NumAromaticRings(_mol_from_smiles(smiles)))
    return f"Aromatic ring count: {v}"

def get_formal_charge(smiles: str) -> str:
    """
    Net formal charge (sum of atom formal charges as encoded in the SMILES).
    RDKit: Chem.GetFormalCharge(mol).
    """
    v = int(Chem.GetFormalCharge(_mol_from_smiles(smiles)))
    return f"Net formal charge: {v}"

def get_qed(smiles: str) -> str:
    """
    QED (Quantitative Estimate of Drug-likeness) as implemented in RDKit.
    RDKit: QED.qed(mol). Returns a score in [0, 1].
    """
    v = float(QED.qed(_mol_from_smiles(smiles)))
    return f"QED (drug-likeness score, 0-1): {v:.4f}"

def get_num_heteroatoms(smiles: str) -> str:
    """Total heteroatom count (non C/H). RDKit: Lipinski.NumHeteroatoms(mol)."""
    v = int(Lipinski.NumHeteroatoms(_mol_from_smiles(smiles)))
    return f"Heteroatom count (non C/H): {v}"

def analyze_ring_systems(smiles: str, standardize: bool = False) -> str:
    """
    Analyze fused ring systems in a molecule.

    Detects ring systems (clusters of rings sharing bonds) and reports their topology,
    aromaticity, and heteroatom content. Distinguishes fused polycyclic systems from
    isolated rings.

    Useful for:
    - AMES mutagenicity: detecting PAH-like systems (>=3 fused aromatic rings)
    - hERG: extended flat aromatic surfaces increase binding risk
    - General structural classification of ring complexity

    Args:
        smiles (str): The SMILES string of the molecule.
        standardize (bool): Standardize SMILES first (remove salts, canonical tautomer).

    Returns:
        str: A formatted string describing the analyzed ring systems, including
             topological, aromaticity, and heteroatom metrics.
    """
    # mol = _validate_smiles(smiles, standardize=standardize)
    mol = Chem.MolFromSmiles(smiles)
    ring_info = mol.GetRingInfo()
    atom_rings: tuple[tuple[int, ...], ...] = ring_info.AtomRings()
    bond_rings: tuple[tuple[int, ...], ...] = ring_info.BondRings()

    if not atom_rings:
        return "No ring systems found."

    # Build ring adjacency: two rings are fused if they share at least one bond
    bond_sets = [set(br) for br in bond_rings]
    n = len(atom_rings)
    adj: dict[int, set[int]] = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if bond_sets[i] & bond_sets[j]:
                adj[i].add(j)
                adj[j].add(i)

    # BFS to find connected components (fused ring systems)
    visited: set[int] = set()
    components: list[list[int]] = []
    for seed in range(n):
        if seed in visited:
            continue
        component: list[int] = []
        queue = [seed]
        while queue:
            node = queue.pop()
            if node in visited:
                continue
            visited.add(node)
            component.append(node)
            queue.extend(adj[node] - visited)
        components.append(component)

    # Analyze each ring system
    # We will format the output lines sequentially
    largest_aromatic = 0
    largest_system = 0
    
    system_descriptions = []

    for component in components:
        system_atoms: set[int] = set()
        ring_sizes: list[int] = []
        aromatic_count = 0

        for ring_idx in component:
            ring_atom_indices = atom_rings[ring_idx]
            system_atoms.update(ring_atom_indices)
            ring_sizes.append(len(ring_atom_indices))
            if all(mol.GetAtomWithIdx(a).GetIsAromatic() for a in ring_atom_indices):
                aromatic_count += 1

        heteroatom_symbols: set[str] = set()
        for atom_idx in system_atoms:
            atom = mol.GetAtomWithIdx(atom_idx)
            if atom.GetAtomicNum() != 6:
                heteroatom_symbols.add(atom.GetSymbol())

        num_rings = len(component)
        largest_system = max(largest_system, num_rings)
        largest_aromatic = max(largest_aromatic, aromatic_count)

        desc = (
            f"  - System has {num_rings} fused ring(s) ({', '.join(map(str, sorted(ring_sizes)))}-membered). "
            f"Aromatic rings: {aromatic_count}. "
            f"Total atoms: {len(system_atoms)}. "
        )
        if heteroatom_symbols:
            desc += f"Heteroatoms: {', '.join(sorted(heteroatom_symbols))}."
        else:
            desc += "No heteroatoms."
        
        system_descriptions.append((num_rings, aromatic_count, desc))

    # Sort largest first by number of rings then aromatic rings
    system_descriptions.sort(key=lambda s: (-s[0], -s[1]))

    # Macrocycle analysis (threshold: 12 atoms)
    all_ring_sizes = [len(ring) for ring in atom_rings]
    largest_ring_size = max(all_ring_sizes) if all_ring_sizes else 0
    num_macrocycles = sum(1 for size in all_ring_sizes if size >= 12)
    
    lines = [
        "Ring System Analysis:",
        f"- Total fused ring systems: {len(components)}",
        f"- Largest fused system size (rings): {largest_system}",
        f"- Largest aromatic system size (rings): {largest_aromatic}",
        f"- Has PAH-like system (>=3 fused aromatic rings): {'Yes' if largest_aromatic >= 3 else 'No'}",
        f"- Largest individual ring size: {largest_ring_size}",
        f"- Has macrocycle (>=12 atoms): {'Yes' if num_macrocycles > 0 else 'No'} (Count: {num_macrocycles})",
        f"- Spiro centers: {rdMolDescriptors.CalcNumSpiroAtoms(mol)}",
        f"- Bridgehead atoms: {rdMolDescriptors.CalcNumBridgeheadAtoms(mol)}",
        "- Systems (Largest first):"
    ]
    
    for _, _, desc in system_descriptions:
        lines.append(desc)

    return "\n".join(lines)

def classify_ionization(
    smiles: str,
    ph: float = 7.4,
    standardize: bool = False,
) -> str:
    """
    Classify the ionization state of a molecule at a target pH.

    Uses Dimorphite-DL to enumerate protonation states, then analyzes the distribution
    of charge states. Returns both summary statistics and a representative variant.

    Args:
        smiles (str): The SMILES string of the molecule.
        ph (float): Target pH for protonation (default: 7.4, physiological).
        standardize (bool): Standardize SMILES first (remove salts only, no tautomer canonicalization).

    Returns:
        str: A formatted string describing the ionization state classification, including
             number of variants, net charges, charge classes, and the most representative state.
    """
    # For ionization, only remove salts (no tautomer canonicalization)
    # mol = _validate_smiles(smiles, standardize=standardize, canonical_tautomer=False)
    mol = Chem.MolFromSmiles(smiles)
    smiles = Chem.MolToSmiles(mol, canonical=True)  # Need SMILES string for Dimorphite

    # Dimorphite-DL may return empty for molecules it can't handle
    try:
        variants = protonate_smiles(smiles, ph_min=ph, ph_max=ph, precision=0.5)
    except Exception:
        variants = []

    # Analyze all variants
    variant_data: list[dict] = []
    for variant_smi in variants:
        mol = Chem.MolFromSmiles(variant_smi)
        if mol is None:
            continue

        pos = neg = 0
        for atom in mol.GetAtoms():
            fc = atom.GetFormalCharge()
            if fc > 0:
                pos += fc
            elif fc < 0:
                neg += abs(fc)

        net = pos - neg
        if pos > 0 and neg > 0:
            charge_class = "zwitterion"
        elif pos > 0:
            charge_class = "base"
        elif neg > 0:
            charge_class = "acid"
        else:
            charge_class = "neutral"

        variant_data.append({"smiles": variant_smi, "net_charge": net, "charge_class": charge_class})

    # If no variants, treat input as neutral
    if not variant_data:
        variant_data = [{"smiles": smiles, "net_charge": 0, "charge_class": "neutral"}]

    # Compute distribution
    
    net_charges = sorted(set(v["net_charge"] for v in variant_data))
    charge_class_counts = Counter(v["charge_class"] for v in variant_data)

    # Pick representative: mode net charge, then lowest |net_charge|
    net_charge_counts = Counter(v["net_charge"] for v in variant_data)
    mode_charge = max(net_charge_counts.keys(), key=lambda c: (net_charge_counts[c], -abs(c)))
    representative_data = next(v for v in variant_data if v["net_charge"] == mode_charge)
    
    # Format the output string
    lines = [
        f"Ionization State Classification (at pH {ph}):",
        f"- Number of protonation states enumerated: {len(variant_data)}",
        f"- Unique net charges observed: {', '.join(map(str, net_charges))}",
        f"- Charge class distribution: {', '.join(f'{k}: {v}' for k, v in charge_class_counts.items())}",
        f"- Has positive states: {'Yes' if any(c > 0 for c in net_charges) else 'No'}",
        f"- Has negative states: {'Yes' if any(c < 0 for c in net_charges) else 'No'}",
        f"- Is ambiguous (pKa near target pH): {'Yes' if len(net_charges) > 1 else 'No'}",
        "- Most representative state:",
        f"  - SMILES: {representative_data['smiles']}",
        f"  - Net charge: {representative_data['net_charge']}",
        f"  - Charge class: {representative_data['charge_class']}"
    ]

    return "\n".join(lines)

# ============================================================
# 1) BASIC tool list (always include)
# ============================================================

classify_ionization_openai = {
    "type": "function",
    "function": {
        "name": "classify_ionization",
        "description": "Classify the ionization state of a molecule at a target pH using Dimorphite-DL. Returns a formatted string with charge distributions.",
        "parameters": {
            "type": "object",
            "properties": {
                "smiles": {
                    "type": "string",
                    "description": "SMILES string of the molecule."
                },
                "ph": {
                    "type": "number",
                    "description": "Target pH for protonation (default: 7.4, physiological)."
                }
            },
            "required": ["smiles"],
        }
    }
}

class AlertLibrary(StrEnum):
    ALL = "all"
    PAINS = "pains"
    PAINS_A = "pains_a"
    PAINS_B = "pains_b"
    PAINS_C = "pains_c"
    BRENK = "brenk"
    NIH = "nih"
    ZINC = "zinc"
    CHEMBL = "chembl"
    CHEMBL_BMS = "chembl_bms"
    CHEMBL_LINT = "chembl_lint"
    CHEMBL_MLSMR = "chembl_mlsmr"

_ALERT_LIBRARY_MAP: dict[AlertLibrary, FilterCatalogParams.FilterCatalogs] = {
    lib: getattr(FilterCatalogParams.FilterCatalogs, lib.name) for lib in AlertLibrary
}

T = TypeVar("T")
def _round_output(value: T) -> T:
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, dict):
        return cast(T, {key: _round_output(item) for key, item in value.items()})
    if isinstance(value, list):
        return cast(T, [_round_output(item) for item in value])
    if isinstance(value, tuple):
        return cast(T, tuple(_round_output(item) for item in value))
    return value

class InvalidFingerprintError(ModelRetry):
    """Exception raised for invalid fingerprint names."""

    pass

class InvalidAlertLibraryError(ModelRetry):
    """Exception raised for invalid structural alert library names."""

    pass

class FingerprintType(StrEnum):
    MORGAN = "morgan"
    RDKIT = "rdkit"
    MACCS = "maccs"
    ATOM_PAIR = "atom_pair"
    TOPOLOGICAL_TORSION = "topological_torsion"

_FINGERPRINT_SIZE = 2048
_MORGAN_RADIUS = 2
_MORGAN_GENERATOR = rfg.GetMorganGenerator(radius=_MORGAN_RADIUS, fpSize=_FINGERPRINT_SIZE, includeChirality=True)
_RDKIT_GENERATOR = rfg.GetRDKitFPGenerator(fpSize=_FINGERPRINT_SIZE)
_ATOM_PAIR_GENERATOR = rfg.GetAtomPairGenerator(fpSize=_FINGERPRINT_SIZE, includeChirality=True)
_TOPOLOGICAL_TORSION_GENERATOR = rfg.GetTopologicalTorsionGenerator(fpSize=_FINGERPRINT_SIZE, includeChirality=True)

_FINGERPRINT_BUILDERS: dict[FingerprintType, Callable[[Chem.Mol], DataStructs.ExplicitBitVect]] = {
    FingerprintType.MORGAN: _MORGAN_GENERATOR.GetFingerprint,
    FingerprintType.RDKIT: _RDKIT_GENERATOR.GetFingerprint,
    FingerprintType.MACCS: MACCSkeys.GenMACCSKeys,  # ty:ignore[unresolved-attribute]
    FingerprintType.ATOM_PAIR: _ATOM_PAIR_GENERATOR.GetFingerprint,
    FingerprintType.TOPOLOGICAL_TORSION: _TOPOLOGICAL_TORSION_GENERATOR.GetFingerprint,
}

class AlertEntry(BaseModel):
    """A single structural alert match."""

    description: str = Field(description="Description of the alert")
    filter_set: str | None = Field(description="Filter set the alert belongs to")
    scope: str | None = Field(description="Scope of the alert (e.g., 'exclude' or 'flag')")

class StructuralAlertsOutput(BaseModel):
    """Output of structural alert screening."""

    library: str = Field(description="Alert library used for screening")
    count: int = Field(description="Total number of alerts matched")
    alerts: list[AlertEntry] = Field(description="List of matched alerts with details")

E = TypeVar("E", bound=StrEnum)
def _coerce_enum(value, enum_cls: type, error_cls: type):
    if isinstance(value, enum_cls):
        return value
    try:
        return enum_cls(value.lower().strip())
    except ValueError as exc:
        raise error_cls(f"Unknown '{value}'. Valid: {', '.join(v.value for v in enum_cls)}") from exc

def score_structural_alerts(
    smiles: str,
    alert_library: AlertLibrary = AlertLibrary.ALL,
    standardize: bool = False,
) -> str:
    """
    Screen a SMILES against RDKit's built-in structural alert catalogs.

    Args:
        smiles (str): Query SMILES.
        alert_library (AlertLibrary): Built-in RDKit library selector.
        standardize (bool): Standardize SMILES first (remove salts, canonical tautomer).

    Returns:
        str: A formatted string describing the matched structural alerts, including
             the library used, total count, and a list of alerts with their details.
    """
    # mol = _validate_smiles(smiles, standardize=standardize)
    mol = Chem.MolFromSmiles(smiles)
    key = _coerce_enum(alert_library, AlertLibrary, InvalidAlertLibraryError)

    params = FilterCatalogParams()
    params.AddCatalog(_ALERT_LIBRARY_MAP[key])
    fc = FilterCatalog(params)

    alerts = []
    for entry in fc.GetMatches(mol):
        props = {name: entry.GetProp(name) for name in entry.GetPropList()}
        alerts.append({
            "description": entry.GetDescription(),
            "filter_set": props.get("FilterSet"),
            "scope": props.get("Scope")
        })

    lines = [
        f"Structural Alert Screening (Library: {key.value}):",
        f"- Total matches: {len(alerts)}"
    ]
    if alerts:
        lines.append("- Alerts found:")
        for alert in alerts:
            parts = [f"Description: {alert['description']}"]
            if alert['filter_set']:
                parts.append(f"FilterSet: {alert['filter_set']}")
            if alert['scope']:
                parts.append(f"Scope: {alert['scope']}")
            lines.append(f"  - " + ", ".join(parts))

    return "\n".join(lines)

score_structural_alerts_openai = {
    "type": "function",
    "function": {
        "name": "score_structural_alerts",
        "description": "Screen a SMILES against RDKit's built-in structural alert catalogs. Returns a formatted string of alerts found.",
        "parameters": {
            "type": "object",
            "properties": {
                "smiles": {
                    "type": "string",
                    "description": "Query SMILES string."
                },
                "alert_library": {
                    "type": "string",
                    "enum": ["brenk", "nih", "chembl", "zinc", "all"],
                    "default": "all",
                    "description": "Which alert library to use. 'all' uses all available libraries."
                }
            },
            "required": ["smiles"],
        }
    }
}

def compute_similarity(
    smiles: str,
    reference_smiles: list[str],
    fingerprint: FingerprintType = FingerprintType.MORGAN,
    standardize: bool = False,
) -> str:
    """
    Compute fingerprint similarity between a query SMILES and a list of reference SMILES.

    Defaults are fixed: Morgan radius=2, 2048 bits, and chirality included.

    Args:
        smiles (str): Query SMILES.
        reference_smiles (list[str]): Reference SMILES to compare against.
        fingerprint (FingerprintType): Fingerprint type.
        standardize (bool): Standardize all SMILES first (remove salts, canonical tautomer).

    Returns:
        str: A formatted string describing the similarity scores between the query SMILES
             and the reference SMILES, sorted descending by similarity.
    """
    # mol = _validate_smiles(smiles, standardize=standardize)
    mol = Chem.MolFromSmiles(smiles)
    refs = [Chem.MolFromSmiles(ref) for ref in reference_smiles]

    fp_key = _coerce_enum(fingerprint, FingerprintType, InvalidFingerprintError)

    def get_fp(m: Chem.Mol) -> DataStructs.ExplicitBitVect:
        return _FINGERPRINT_BUILDERS[fp_key](m)

    qfp = get_fp(mol)
    ref_fps = [get_fp(m) for m in refs]

    # C++ bulk compute (much faster than Python loop)
    sims = DataStructs.BulkTanimotoSimilarity(qfp, ref_fps)

    similarities = [
        {"reference_smiles": ref_smi, "similarity": _round_output(float(sim))}
        for ref_smi, sim in zip(reference_smiles, sims, strict=False)
    ]
    similarities.sort(key=lambda x: x["similarity"], reverse=True)

    lines = [
        f"Fingerprint Similarity Analysis (Method: {fp_key.value}):",
        f"Query SMILES: {smiles}",
        "Top similarities (sorted descending):"
    ]
    
    for entry in similarities:
        lines.append(f"  - {entry['reference_smiles']}: {entry['similarity']:.4f}")
        
    return "\n".join(lines)

compute_similarity_openai = {
        "type": "function",
        "function": {
            "name": "compute_similarity",
            "description": "Compute fingerprint similarity (Morgan by default) between a query SMILES and a list of reference SMILES. Returns a formatted string.",
            "parameters": {
                "type": "object",
                "properties": {
                    "smiles": {
                        "type": "string",
                        "description": "Query SMILES string."
                    },
                    "reference_smiles": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of reference SMILES strings to compare against."
                    }
                },
                "required": ["smiles", "reference_smiles"],
            }
        }
    }
_FEATURE_FACTORY = None
def _get_feature_factory() -> "ChemicalFeatures.MolChemicalFeatureFactory":
    global _FEATURE_FACTORY
    if _FEATURE_FACTORY is None:
        fdef = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
        _FEATURE_FACTORY = ChemicalFeatures.BuildFeatureFactory(fdef)  # ty:ignore[unresolved-attribute]
    return _FEATURE_FACTORY

def extract_pharmacophore_features(smiles: str, standardize: bool = False) -> str:
    """
    Extract pharmacophore-like features using RDKit's BaseFeatures definitions.

    Args:
        smiles (str): Query SMILES.
        standardize (bool): Standardize SMILES first (remove salts, canonical tautomer).

    Returns:
        str: A formatted string describing the extracted pharmacophore features.
             Includes feature counts and a detailed list of features with family, type, and atom_ids.
    """
    # mol = _validate_smiles(smiles, standardize=standardize)
    mol = Chem.MolFromSmiles(smiles)
    factory = _get_feature_factory()
    features = factory.GetFeaturesForMol(mol)

    counts = Counter()
    items = []
    for feat in features:
        family = feat.GetFamily()
        counts[family] += 1
        items.append({
            "family": family,
            "type": feat.GetType(),
            "atom_ids": list(feat.GetAtomIds()),
        })

    output = "Pharmacophore Feature Extraction Results:\n"
    output += "- feature_counts:\n"
    if counts:
        for k, v in counts.items():
            output += f"  - {k}: {v}\n"
    else:
        output += "  (None)\n"
        
    output += "- features:\n"
    if items:
        for item in items:
            output += f"  - family: {item['family']}, type: {item['type']}, atom_ids: {item['atom_ids']}\n"
    else:
        output += "  (None)\n"

    return output.strip()

RDKIT_BASIC_OPENAI_TOOLS = [
    _tool("get_molecular_weight", "Return the average molecular weight (Daltons)."),
    _tool("get_exact_molecular_weight", "Return the monoisotopic exact molecular weight (Daltons)."),
    _tool("get_heavy_atom_count", "Return the number of non-hydrogen atoms (heavy atoms)."),
    _tool("get_mol_logp", "Return Wildman–Crippen cLogP (octanol/water)."),
    _tool("get_tpsa", "Return topological polar surface area TPSA (Å^2)."),
    _tool("get_hbd", "Return Lipinski H-bond donors count."),
    _tool("get_hba", "Return Lipinski H-bond acceptors count."),
    _tool("get_num_rotatable_bonds", "Return rotatable bonds count."),
    _tool("get_fraction_csp3", "Return FractionCSP3 = (# sp3 carbon atoms)/(# total carbon atoms), Often used as a simple “3D character / saturation” proxy from 2D structure."),
    _tool("get_mol_mr", "Return Wildman–Crippen molar refractivity (MR)."),
    _tool("get_ring_count", "Return total ring count."),
    _tool("get_num_aromatic_rings", "Return aromatic ring count."),
    _tool("get_formal_charge", "Return net formal charge (sum of atom formal charges as encoded in the SMILES)."),
    _tool("get_qed", "Return QED (Quantitative Estimate of Drug-likeness) as implemented in RDKit."),
    _tool("get_num_heteroatoms", "Return heteroatom count (non C/H)."),
    _tool("analyze_ring_systems", "Analyze fused ring systems (clusters of rings sharing bonds) and report their topology, aromaticity, and heteroatom content. Distinguishes fused polycyclic systems from isolated rings. Useful for AMES mutagenicity (detecting PAH-like systems), hERG (extended flat aromatic surfaces), and general structural classification of ring complexity."),
    classify_ionization_openai,
    compute_similarity_openai,
    score_structural_alerts_openai,
    _tool("extract_pharmacophore_features", "Extract pharmacophore features using RDKit's BaseFeatures definitions."),
]

# ============================================================
# B) “SPECIFIC” building blocks (no fragments, still SMILES-only)
#    These get mixed into task-specific packs below.
# ============================================================

# --- Permeability / surface proxy ---
def get_labute_asa(smiles: str) -> str:
    """
    Labute approximate surface area (ASA proxy; topology/fragment-based, not true 3D SASA).
    RDKit: rdMolDescriptors.CalcLabuteASA(mol).
    """
    v = float(rdMolDescriptors.CalcLabuteASA(_mol_from_smiles(smiles)))
    return f"Labute approximate surface area (ASA proxy): {v:.4f}"

# --- Charge distribution proxy (Gasteiger; SMILES-only) ---
def get_max_abs_partial_charge(smiles: str) -> str:
    """Maximum absolute Gasteiger partial charge among atoms. RDKit: Descriptors.MaxAbsPartialCharge(mol)."""
    v = float(Descriptors.MaxAbsPartialCharge(_mol_from_smiles(smiles)))
    return f"Max absolute Gasteiger partial charge: {v:.4f}"

def get_min_abs_partial_charge(smiles: str) -> str:
    """Minimum absolute Gasteiger partial charge among atoms. RDKit: Descriptors.MinAbsPartialCharge(mol)."""
    v = float(Descriptors.MinAbsPartialCharge(_mol_from_smiles(smiles)))
    return f"Min absolute Gasteiger partial charge: {v:.4f}"

# --- EState extremes (electrotopological environment proxies) ---
def get_max_estate_index(smiles: str) -> str:
    """Maximum EState index among atoms. RDKit: Descriptors.MaxEStateIndex(mol)."""
    v = float(Descriptors.MaxEStateIndex(_mol_from_smiles(smiles)))
    return f"Max EState index: {v:.4f}"

def get_min_estate_index(smiles: str) -> str:
    """Minimum EState index among atoms. RDKit: Descriptors.MinEStateIndex(mol)."""
    v = float(Descriptors.MinEStateIndex(_mol_from_smiles(smiles)))
    return f"Min EState index: {v:.4f}"

# --- Aromaticity (atom-level) ---
def get_num_aromatic_atoms(smiles: str) -> str:
    """Number of aromatic atoms. RDKit: sum(atom.GetIsAromatic() for atoms)."""
    mol = _mol_from_smiles(smiles)
    v = int(sum(1 for a in mol.GetAtoms() if a.GetIsAromatic()))
    return f"Aromatic atom count: {v}"

def get_fraction_aromatic_atoms(smiles: str) -> str:
    """Fraction aromatic atoms = (# aromatic atoms)/(# atoms). RDKit aromaticity flags."""
    mol = _mol_from_smiles(smiles)
    n = mol.GetNumAtoms()
    v = (sum(1 for a in mol.GetAtoms() if a.GetIsAromatic()) / n) if n else 0.0
    return f"Fraction aromatic atoms: {float(v):.4f}"

# --- Formal-charge centers (helps hERG/P-gp/BBB behavior) ---
def get_num_positive_charge_atoms(smiles: str) -> str:
    """Count atoms with formal charge > 0 (cationic centers). RDKit: atom.GetFormalCharge()."""
    mol = _mol_from_smiles(smiles)
    v = int(sum(1 for a in mol.GetAtoms() if a.GetFormalCharge() > 0))
    return f"Cationic center count (formal charge > 0): {v}"

def get_num_negative_charge_atoms(smiles: str) -> str:
    """Count atoms with formal charge < 0 (anionic centers). RDKit: atom.GetFormalCharge()."""
    mol = _mol_from_smiles(smiles)
    v = int(sum(1 for a in mol.GetAtoms() if a.GetFormalCharge() < 0))
    return f"Anionic center count (formal charge < 0): {v}"

# --- Ring type splits (useful for BBB/hERG/CYP) ---
def get_num_aliphatic_rings(smiles: str) -> str:
    """Aliphatic ring count. RDKit: Lipinski.NumAliphaticRings(mol)."""
    v = int(Lipinski.NumAliphaticRings(_mol_from_smiles(smiles)))
    return f"Aliphatic ring count: {v}"

def get_num_saturated_rings(smiles: str) -> str:
    """Saturated ring count. RDKit: Lipinski.NumSaturatedRings(mol)."""
    v = int(Lipinski.NumSaturatedRings(_mol_from_smiles(smiles)))
    return f"Saturated ring count: {v}"

def get_num_heterocycles(smiles: str) -> str:
    """Heterocycle count. RDKit: Lipinski.NumHeterocycles(mol)."""
    v = int(Lipinski.NumHeterocycles(_mol_from_smiles(smiles)))
    return f"Heterocycle count: {v}"

def get_num_aromatic_heterocycles(smiles: str) -> str:
    """Aromatic heterocycle count. RDKit: Lipinski.NumAromaticHeterocycles(mol)."""
    v = int(Lipinski.NumAromaticHeterocycles(_mol_from_smiles(smiles)))
    return f"Aromatic heterocycle count: {v}"

def get_num_aliphatic_heterocycles(smiles: str) -> str:
    """Aliphatic heterocycle count. RDKit: Lipinski.NumAliphaticHeterocycles(mol)."""
    v = int(Lipinski.NumAliphaticHeterocycles(_mol_from_smiles(smiles)))
    return f"Aliphatic heterocycle count: {v}"

def get_num_saturated_heterocycles(smiles: str) -> str:
    """Saturated heterocycle count. RDKit: Lipinski.NumSaturatedHeterocycles(mol)."""
    v = int(Lipinski.NumSaturatedHeterocycles(_mol_from_smiles(smiles)))
    return f"Saturated heterocycle count: {v}"

def get_num_amide_bonds(smiles: str) -> str:
    """Amide bond count (Lipinski). RDKit: Lipinski.NumAmideBonds(mol)."""
    v = int(Lipinski.NumAmideBonds(_mol_from_smiles(smiles)))
    return f"Amide bond count: {v}"

# --- Topological/complexity pack (broad tox panels love these) ---
def get_bertz_ct(smiles: str) -> str:
    """Bertz complexity index. RDKit: Descriptors.BertzCT(mol)."""
    v = float(Descriptors.BertzCT(_mol_from_smiles(smiles)))
    return f"Bertz complexity index: {v:.4f}"

def get_balaban_j(smiles: str) -> str:
    """Balaban J topological index. RDKit: Descriptors.BalabanJ(mol)."""
    v = float(Descriptors.BalabanJ(_mol_from_smiles(smiles)))
    return f"Balaban J index: {v:.4f}"

def get_ipc(smiles: str) -> str:
    """Information content (IPC). RDKit: Descriptors.Ipc(mol)."""
    v = float(Descriptors.Ipc(_mol_from_smiles(smiles)))
    return f"IPC (information content): {v:.4f}"

def get_hall_kier_alpha(smiles: str) -> str:
    """Hall–Kier alpha. RDKit: Descriptors.HallKierAlpha(mol)."""
    v = float(Descriptors.HallKierAlpha(_mol_from_smiles(smiles)))
    return f"Hall–Kier alpha: {v:.4f}"

def get_kappa1(smiles: str) -> str:
    """Kappa1 shape index. RDKit: Descriptors.Kappa1(mol)."""
    v = float(Descriptors.Kappa1(_mol_from_smiles(smiles)))
    return f"Kappa1 shape index: {v:.4f}"

def get_kappa2(smiles: str) -> str:
    """Kappa2 shape index. RDKit: Descriptors.Kappa2(mol)."""
    v = float(Descriptors.Kappa2(_mol_from_smiles(smiles)))
    return f"Kappa2 shape index: {v:.4f}"

def get_kappa3(smiles: str) -> str:
    """Kappa3 shape index. RDKit: Descriptors.Kappa3(mol)."""
    v = float(Descriptors.Kappa3(_mol_from_smiles(smiles)))
    return f"Kappa3 shape index: {v:.4f}"

# --- Stereo (sometimes helps ClinTox/DILI and broad tasks) ---
def get_num_atom_stereo_centers(smiles: str) -> str:
    """Number of atom stereocenters. RDKit: rdMolDescriptors.CalcNumAtomStereoCenters(mol)."""
    v = int(rdMolDescriptors.CalcNumAtomStereoCenters(_mol_from_smiles(smiles)))
    return f"Atom stereocenter count: {v}"

def get_num_unspecified_atom_stereo_centers(smiles: str) -> str:
    """Number of unspecified atom stereocenters. RDKit: rdMolDescriptors.CalcNumUnspecifiedAtomStereoCenters(mol)."""
    v = int(rdMolDescriptors.CalcNumUnspecifiedAtomStereoCenters(_mol_from_smiles(smiles)))
    return f"Unspecified atom stereocenter count: {v}"

def get_esol(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    logP = Crippen.MolLogP(mol)
    MW = Descriptors.MolWt(mol)
    RB = Lipinski.NumRotatableBonds(mol)

    heavy_atoms = mol.GetNumHeavyAtoms()
    aromatic_atoms = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
    AP = aromatic_atoms / heavy_atoms if heavy_atoms > 0 else 0

    logS = 0.16 - 0.63 * logP - 0.0062 * MW + 0.066 * RB - 0.74 * AP
    return f"ESOL logS: {logS:.4f}"

# ============================================================
# 2) Task-specific packs (add on top of BASIC)
#    (No fragments; relaxed but still interpretable.)
# ============================================================

# ---- hERG (hERG, hERG_Karim, herg_central) ----
RDKIT_HERG_OPENAI_TOOLS: List[Dict[str, Any]] = [
    _tool("get_num_positive_charge_atoms", "Return count of cationic centers (atoms with formal charge > 0)."),
    _tool("get_num_negative_charge_atoms", "Return count of anionic centers (atoms with formal charge < 0)."),
    _tool("get_fraction_aromatic_atoms", "Return fraction of aromatic atoms (aromatic atoms / total atoms)."),
    _tool("get_num_aromatic_atoms", "Return number of aromatic atoms."),
    _tool("get_labute_asa", "Return Labute approximate surface area (ASA proxy) using RDKit rdMolDescriptors.CalcLabuteASA."),
    _tool("get_num_amide_bonds", "Return amide bond count using RDKit Lipinski.NumAmideBonds."),
]

# ---- Permeability / absorption (PAMPA, HIA, Bioavailability) ----
RDKIT_PERMEABILITY_OPENAI_TOOLS: List[Dict[str, Any]] = [
    _tool("get_labute_asa", "Return Labute approximate surface area (ASA proxy) using RDKit rdMolDescriptors.CalcLabuteASA."),
    _tool("get_fraction_aromatic_atoms", "Return fraction of aromatic atoms (aromatic atoms / total atoms)."),
    _tool("get_num_positive_charge_atoms", "Return count of cationic centers (atoms with formal charge > 0)."),
    _tool("get_num_negative_charge_atoms", "Return count of anionic centers (atoms with formal charge < 0)."),
]

# ---- BBB (BBB_Martins) ----
RDKIT_BBB_OPENAI_TOOLS: List[Dict[str, Any]] = [
    _tool("get_labute_asa", "Return Labute approximate surface area (ASA proxy)."),
    _tool("get_fraction_aromatic_atoms", "Return fraction of aromatic atoms (aromatic atoms / total atoms)."),
    _tool("get_num_aromatic_atoms", "Return number of aromatic atoms."),
    _tool("get_num_aliphatic_rings", "Return aliphatic ring count using RDKit Lipinski.NumAliphaticRings."),
    _tool("get_num_saturated_rings", "Return saturated ring count using RDKit Lipinski.NumSaturatedRings."),
    _tool("get_num_positive_charge_atoms", "Return count of cationic centers (atoms with formal charge > 0)."),
    _tool("get_num_negative_charge_atoms", "Return count of anionic centers (atoms with formal charge < 0)."),
]

# ---- P-gp (Pgp_Broccatelli) ----
RDKIT_PGP_OPENAI_TOOLS: List[Dict[str, Any]] = [
    _tool("get_labute_asa", "Return Labute approximate surface area (ASA proxy)."),
    _tool("get_fraction_aromatic_atoms", "Return fraction of aromatic atoms."),
    _tool("get_num_positive_charge_atoms", "Return count of cationic centers (atoms with formal charge > 0)."),
    _tool("get_num_negative_charge_atoms", "Return count of anionic centers (atoms with formal charge < 0)."),
    _tool("get_num_amide_bonds", "Return amide bond count."),
]

# ---- CYP inhibition (CYP*_Veith) ----
RDKIT_CYP_INHIB_OPENAI_TOOLS: List[Dict[str, Any]] = [
    _tool("get_num_heterocycles", "Return heterocycle count using RDKit Lipinski.NumHeterocycles."),
    _tool("get_num_aromatic_heterocycles", "Return aromatic heterocycle count using RDKit Lipinski.NumAromaticHeterocycles."),
    _tool("get_num_aliphatic_heterocycles", "Return aliphatic heterocycle count using RDKit Lipinski.NumAliphaticHeterocycles."),
    _tool("get_num_saturated_heterocycles", "Return saturated heterocycle count using RDKit Lipinski.NumSaturatedHeterocycles."),
    _tool("get_num_amide_bonds", "Return amide bond count using RDKit Lipinski.NumAmideBonds."),
    _tool("get_fraction_aromatic_atoms", "Return fraction of aromatic atoms."),
]

# ---- CYP substrate (CYP*_Substrate_CarbonMangels) ----
RDKIT_CYP_SUBSTRATE_OPENAI_TOOLS: List[Dict[str, Any]] = [
    _tool("get_num_heterocycles", "Return heterocycle count."),
    _tool("get_num_aromatic_heterocycles", "Return aromatic heterocycle count."),
    _tool("get_num_amide_bonds", "Return amide bond count."),
    _tool("get_labute_asa", "Return Labute approximate surface area (ASA proxy)."),
    _tool("get_fraction_aromatic_atoms", "Return fraction of aromatic atoms."),
    _tool("get_num_atom_stereo_centers", "Return atom stereocenter count."),
    _tool("get_num_unspecified_atom_stereo_centers", "Return unspecified stereocenter count."),
]

# ---- Genotoxicity / carcinogenicity (AMES, Carcinogens_Lagunin) ----
RDKIT_GENOTOX_OPENAI_TOOLS: List[Dict[str, Any]] = [
    _tool("get_fraction_aromatic_atoms", "Return fraction of aromatic atoms."),
    _tool("get_num_aromatic_atoms", "Return aromatic atom count."),
    _tool("get_max_abs_partial_charge", "Return max absolute Gasteiger partial charge."),
    _tool("get_min_abs_partial_charge", "Return min absolute Gasteiger partial charge."),
    _tool("get_max_estate_index", "Return max EState index among atoms."),
    _tool("get_min_estate_index", "Return min EState index among atoms."),
    _tool("get_bertz_ct", "Return Bertz complexity index."),
    _tool("get_balaban_j", "Return Balaban J topological index."),
]

# ---- DILI / ClinTox (systemic tox; broad, tolerant) ----
RDKIT_SYSTEMIC_TOX_OPENAI_TOOLS: List[Dict[str, Any]] = [
    _tool("get_labute_asa", "Return Labute approximate surface area (ASA proxy)."),
    _tool("get_bertz_ct", "Return Bertz complexity index."),
    _tool("get_ipc", "Return IPC (information content)."),
    _tool("get_hall_kier_alpha", "Return Hall–Kier alpha."),
    _tool("get_kappa1", "Return Kappa1 shape index."),
    _tool("get_kappa2", "Return Kappa2 shape index."),
    _tool("get_kappa3", "Return Kappa3 shape index."),
    _tool("get_num_atom_stereo_centers", "Return atom stereocenter count."),
    _tool("get_num_unspecified_atom_stereo_centers", "Return unspecified stereocenter count."),
    _tool("get_esol", "Return estimated ESOL logS (solubility proxy)."),
]

# ---- Skin sensitization / Skin Reaction (permeability + reactivity proxies) ----
RDKIT_SKIN_REACTION_OPENAI_TOOLS: List[Dict[str, Any]] = [
    _tool("get_labute_asa", "Return Labute approximate surface area (ASA proxy)."),
    _tool("get_num_positive_charge_atoms", "Return count of cationic centers (atoms with formal charge > 0)."),
    _tool("get_num_negative_charge_atoms", "Return count of anionic centers (atoms with formal charge < 0)."),
    _tool("get_max_abs_partial_charge", "Return max absolute Gasteiger partial charge."),
    _tool("get_min_abs_partial_charge", "Return min absolute Gasteiger partial charge."),
    _tool("get_max_estate_index", "Return max EState index among atoms."),
    _tool("get_min_estate_index", "Return min EState index among atoms."),
]

# ---- Broad tox panels (Tox21, ToxCast, many “misc tox” sets) ----
RDKIT_BROAD_TOX_PANEL_OPENAI_TOOLS: List[Dict[str, Any]] = [
    _tool("get_labute_asa", "Return Labute approximate surface area (ASA proxy)."),
    _tool("get_fraction_aromatic_atoms", "Return fraction of aromatic atoms."),
    _tool("get_max_abs_partial_charge", "Return max absolute Gasteiger partial charge."),
    _tool("get_min_abs_partial_charge", "Return min absolute Gasteiger partial charge."),
    _tool("get_max_estate_index", "Return max EState index among atoms."),
    _tool("get_min_estate_index", "Return min EState index among atoms."),
    _tool("get_bertz_ct", "Return Bertz complexity index."),
    _tool("get_balaban_j", "Return Balaban J topological index."),
    _tool("get_ipc", "Return IPC (information content)."),
    _tool("get_kappa1", "Return Kappa1 shape index."),
    _tool("get_kappa2", "Return Kappa2 shape index."),
    _tool("get_kappa3", "Return Kappa3 shape index."),
]

# ---- Viral activity (HIV, SARSCoV2 assays): general “binding-ish” chemistry pack ----
RDKIT_ANTIVIRAL_ACTIVITY_OPENAI_TOOLS: List[Dict[str, Any]] = [
    _tool("get_labute_asa", "Return Labute approximate surface area (ASA proxy)."),
    _tool("get_fraction_aromatic_atoms", "Return fraction of aromatic atoms."),
    _tool("get_num_heterocycles", "Return heterocycle count."),
    _tool("get_num_aromatic_heterocycles", "Return aromatic heterocycle count."),
    _tool("get_num_atom_stereo_centers", "Return atom stereocenter count."),
    _tool("get_bertz_ct", "Return Bertz complexity index."),
    _tool("get_kappa1", "Return Kappa1 shape index."),
    _tool("get_kappa2", "Return Kappa2 shape index."),
]


# ============================================================
# 3) Per-task mapping (exact keys you listed)
# ============================================================
TDC_RDKIT_SPECIFIC_OPENAI_TOOLS_MAP: Dict[str, List[Dict[str, Any]]] = {
    # hERG family
    "hERG_Karim": RDKIT_HERG_OPENAI_TOOLS,
    "hERG": RDKIT_HERG_OPENAI_TOOLS,
    "herg_central_hERG_inhib": RDKIT_HERG_OPENAI_TOOLS,

    # Genotox / carcinogenicity
    "Carcinogens_Lagunin": RDKIT_GENOTOX_OPENAI_TOOLS,
    "AMES": RDKIT_GENOTOX_OPENAI_TOOLS,

    # Systemic tox
    "DILI": RDKIT_SYSTEMIC_TOX_OPENAI_TOOLS,
    "ClinTox": RDKIT_SYSTEMIC_TOX_OPENAI_TOOLS,

    # Skin sensitization
    "Skin_Reaction": RDKIT_SKIN_REACTION_OPENAI_TOOLS,

    # Broad panels
    "ToxCast": RDKIT_BROAD_TOX_PANEL_OPENAI_TOOLS,
    "Tox21": RDKIT_BROAD_TOX_PANEL_OPENAI_TOOLS,
    "butkiewicz": RDKIT_BROAD_TOX_PANEL_OPENAI_TOOLS,   # (kept broad; name is ambiguous across benchmarks)

    # Permeability / absorption / distribution
    "PAMPA_NCATS": RDKIT_PERMEABILITY_OPENAI_TOOLS,
    "HIA_Hou": RDKIT_PERMEABILITY_OPENAI_TOOLS,
    "Bioavailability_Ma": RDKIT_PERMEABILITY_OPENAI_TOOLS,
    "BBB_Martins": RDKIT_BBB_OPENAI_TOOLS,
    "Pgp_Broccatelli": RDKIT_PGP_OPENAI_TOOLS,

    # CYP inhibition (Veith)
    "CYP1A2_Veith": RDKIT_CYP_INHIB_OPENAI_TOOLS,
    "CYP2C19_Veith": RDKIT_CYP_INHIB_OPENAI_TOOLS,
    "CYP2C9_Veith": RDKIT_CYP_INHIB_OPENAI_TOOLS,
    "CYP2D6_Veith": RDKIT_CYP_INHIB_OPENAI_TOOLS,
    "CYP3A4_Veith": RDKIT_CYP_INHIB_OPENAI_TOOLS,

    # CYP substrate (CarbonMangels)
    "CYP2C9_Substrate_CarbonMangels": RDKIT_CYP_SUBSTRATE_OPENAI_TOOLS,
    "CYP2D6_Substrate_CarbonMangels": RDKIT_CYP_SUBSTRATE_OPENAI_TOOLS,
    "CYP3A4_Substrate_CarbonMangels": RDKIT_CYP_SUBSTRATE_OPENAI_TOOLS,

    # Viral / bioactivity assays
    "HIV": RDKIT_ANTIVIRAL_ACTIVITY_OPENAI_TOOLS,
    "SARSCoV2_3CLPro_Diamond": RDKIT_ANTIVIRAL_ACTIVITY_OPENAI_TOOLS,
    "SARSCoV2_Vitro_Touret": RDKIT_ANTIVIRAL_ACTIVITY_OPENAI_TOOLS,

    # (If these are not SMILES-based, keep them empty or map to a minimal pack)
    "SAbDab_Chen": [],  # Often antibody/protein-centric; keep empty unless you confirm SMILES inputs exist.
}

# -------------------------
# All RDKit descriptors
# -------------------------
def calc_all_rdkit_descriptors(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")

    results = {}
    errors = {}

    # 1) Descriptors._descList（最核心）
    for name, fn in Descriptors._descList:
        try:
            results[name] = fn(mol)
        except Exception as e:
            errors[name] = str(e)

    # 2) rdMolDescriptors: Calc* 系列
    for name in [n for n in dir(rdMolDescriptors) if n.startswith("Calc")]:
        fn = getattr(rdMolDescriptors, name)
        if callable(fn):
            try:
                results[name] = fn(mol)
            except Exception as e:
                errors[name] = str(e)

    # 3) Lipinski: 常见计数
    for name in [n for n in dir(Lipinski) if n.startswith("Num") or n.startswith("Calc")]:
        fn = getattr(Lipinski, name)
        if callable(fn):
            try:
                results[f"Lipinski.{name}"] = fn(mol)
            except Exception as e:
                errors[f"Lipinski.{name}"] = str(e)

    # 4) Crippen
    for name in [n for n in dir(Crippen) if n.startswith("Mol")]:
        fn = getattr(Crippen, name)
        if callable(fn):
            try:
                results[f"Crippen.{name}"] = fn(mol)
            except Exception as e:
                errors[f"Crippen.{name}"] = str(e)

    # 5) GraphDescriptors（有些函数名不是 Calc 开头）
    for name in [n for n in dir(GraphDescriptors) if n and n[0].isupper()]:
        fn = getattr(GraphDescriptors, name)
        if callable(fn):
            try:
                results[f"Graph.{name}"] = fn(mol)
            except Exception as e:
                errors[f"Graph.{name}"] = str(e)

    # 6) Fragments: fr_* 计数
    for name in [n for n in dir(Fragments) if n.startswith("fr_")]:
        fn = getattr(Fragments, name)
        if callable(fn):
            try:
                results[f"Frag.{name}"] = fn(mol)
            except Exception as e:
                errors[f"Frag.{name}"] = str(e)

    return results, errors

if __name__ == "__main__":
    smiles = "CCOC(=O)C(=NOC(C)(C)C(=O)OC(C)(C)C)c1csc(NC(c2ccccc2)(c2ccccc2)c2ccccc2)n1"
    # results, errors = calc_all_rdkit_descriptors(smiles)
    print(get_esol(smiles))
    # print(errors)
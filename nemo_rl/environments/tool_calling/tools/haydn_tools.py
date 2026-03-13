import io
import math
import os
from collections import Counter
from collections.abc import Callable
# from contextlib import redirect_stdout
from enum import StrEnum
from typing import Any, cast

from pydantic import BaseModel, Field
from pydantic_ai import ModelRetry
from rdkit import Chem, DataStructs, RDConfig, RDLogger
from rdkit.Chem import (
    QED,
    ChemicalFeatures,
    Descriptors,
    Lipinski,
    MACCSkeys,
    rdFingerprintGenerator as rfg,
    rdFMCS,
    rdMolDescriptors,
)
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")  # ty:ignore[unresolved-attribute]


class InvalidFingerprintError(ModelRetry):
    """Exception raised for invalid fingerprint names."""

    pass


class InvalidAlertLibraryError(ModelRetry):
    """Exception raised for invalid structural alert library names."""

    pass


class InvalidSMILESError(ModelRetry):
    """Exception raised for invalid SMILES strings."""

    pass


class InvalidDescriptorError(ModelRetry):
    """Exception raised for invalid descriptor names."""

    pass


class FingerprintType(StrEnum):
    MORGAN = "morgan"
    RDKIT = "rdkit"
    MACCS = "maccs"
    ATOM_PAIR = "atom_pair"
    TOPOLOGICAL_TORSION = "topological_torsion"


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


class DescriptorName(StrEnum):
    MASSES = "masses"
    ATOM_COUNTS = "atom_counts"
    SURFACE_SHAPE_PROPS = "surface_shape_props"
    RING_COUNTS = "ring_counts"
    LOGP = "logp"
    NUM_ROTATABLE_BONDS = "num_rotatable_bonds"
    NUM_AMIDE_BONDS = "num_amide_bonds"
    FORMAL_CHARGE = "formal_charge"
    QED = "qed"
    HYDROGEN_BONDING = "hydrogen_bonding"
    LIPINSKI_VIOLATIONS = "lipinski_violations"
    ESOL = "esol"


################################################################################
# Output Models
################################################################################


class SimilarityEntry(BaseModel):
    """Similarity score for a single reference molecule."""

    reference_smiles: str = Field(description="The reference SMILES string")
    similarity: float = Field(description="Tanimoto similarity score (0-1)")


class SimilarityOutput(BaseModel):
    """Output of fingerprint similarity computation."""

    fingerprint: str = Field(description="Fingerprint type used for comparison")
    similarities: list[SimilarityEntry] = Field(description="Similarity scores sorted descending")


class MCSOutput(BaseModel):
    """Output of maximum common substructure search."""

    smarts: str = Field(description="SMARTS pattern of the MCS")
    smiles: str | None = Field(description="SMILES representation of the MCS (if convertible)")
    num_atoms: int = Field(description="Number of atoms in the MCS")
    num_bonds: int = Field(description="Number of bonds in the MCS")
    canceled: bool = Field(description="Whether the search was terminated due to timeout")
    query_coverage: float = Field(description="Fraction of query atoms covered by MCS")
    ref_coverages: list[float] = Field(description="Fraction of each reference covered by MCS")


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


class PharmacophoreFeature(BaseModel):
    """A single pharmacophore feature."""

    family: str = Field(description="Feature family (e.g., 'Donor', 'Acceptor', 'Aromatic')")
    type: str = Field(description="Specific feature type")
    atom_ids: list[int] = Field(description="Atom indices involved in this feature")


class PharmacophoreOutput(BaseModel):
    """Output of pharmacophore feature extraction."""

    feature_counts: dict[str, int] = Field(description="Count of features per family")
    features: list[PharmacophoreFeature] = Field(description="List of all extracted features")


class IonizationRepresentative(BaseModel):
    """Representative protonation state."""

    smiles: str = Field(description="SMILES of the representative protonation state")
    net_charge: int = Field(description="Net formal charge of this state")
    charge_class: str = Field(description="Classification: 'acid', 'base', 'zwitterion', or 'neutral'")


class IonizationOutput(BaseModel):
    """Output of ionization state classification."""

    num_variants: int = Field(description="Number of protonation states enumerated")
    net_charges: list[int] = Field(description="Sorted list of unique net charges observed")
    charge_class_counts: dict[str, int] = Field(description="Count per charge class (acid/base/zwitterion/neutral)")
    has_positive_states: bool = Field(description="Whether any variant has net positive charge")
    has_negative_states: bool = Field(description="Whether any variant has net negative charge")
    is_ambiguous: bool = Field(description="Whether multiple net charge states exist (pKa near target pH)")
    representative: IonizationRepresentative = Field(description="Most representative protonation state")


class SubstructureMatchResult(BaseModel):
    """Result of a single substructure pattern match."""

    present: bool = Field(description="Whether the pattern was found in the molecule")
    count: int = Field(description="Number of non-overlapping matches")


class RingSystemInfo(BaseModel):
    """Information about a single fused ring system."""

    num_rings: int = Field(description="SSSR ring count in this system")
    num_aromatic: int = Field(description="Fully aromatic ring count")
    all_aromatic: bool = Field(description="Whether every ring in the system is aromatic")
    atom_count: int = Field(description="Unique atoms in the system")
    ring_sizes: list[int] = Field(description="Individual ring sizes (sorted)")
    heteroatoms: list[str] = Field(description="Sorted heteroatom symbols (empty = all-carbon)")


class RingSystemsOutput(BaseModel):
    """Output of ring system analysis."""

    ring_systems: list[RingSystemInfo] = Field(description="List of fused ring systems (largest first)")
    num_ring_systems: int = Field(description="Total fused-system count")
    largest_system_size: int = Field(description="Max rings in any one system")
    largest_aromatic_system: int = Field(description="Max aromatic rings in any one system")
    has_pah_like: bool = Field(description="Whether any system has >=3 fused aromatic rings (PAH-like)")
    largest_ring_size: int = Field(description="Size of the largest individual ring")
    has_macrocycle: bool = Field(description="Whether any ring has >=12 atoms")
    num_macrocycles: int = Field(description="Count of rings with >=12 atoms")
    spiro_atoms: int = Field(description="Number of spiro centers")
    bridgehead_atoms: int = Field(description="Number of bridgehead atoms")


class MurckoScaffoldOutput(BaseModel):
    """Output of Murcko scaffold extraction."""

    scaffold_smiles: str = Field(description="SMILES of the Murcko scaffold (empty if no scaffold)")
    generic_scaffold_smiles: str | None = Field(description="Generic scaffold SMILES (if requested)")
    num_scaffold_atoms: int = Field(description="Heavy atom count in scaffold")
    num_scaffold_rings: int = Field(description="Ring count in scaffold")
    scaffold_fraction: float = Field(description="Scaffold atoms / molecule heavy atoms")


class pKaPredictionOutput(BaseModel):
    base_sites: dict[int, float] = Field(description="Base-site pKa values (1-indexed atom map numbers)")

    acid_sites: dict[int, float] = Field(description="Acid-site pKa values (1-indexed atom map numbers)")

    most_basic_pka: float | None = Field(
        description="Maximum predicted base-site pKa across the molecule (None if no base sites are predicted)"
    )

    most_acidic_pka: float | None = Field(
        description="Minimum predicted acid-site pKa across the molecule (None if no acid sites are predicted)"
    )

    num_basic_sites: int = Field(description="Number of predicted base (protonatable) sites")

    num_acidic_sites: int = Field(description="Number of predicted acidic (deprotonatable) sites")

    mapped_smiles: str = Field(description="SMILES of the protonated molecule with atom map numbers set")


class LogDEstimateOutput(BaseModel):
    """Estimated logD and supporting signals at a target pH."""

    logd: float = Field(description="Estimated logD at the specified pH (approximate)")
    logp: float = Field(description="RDKit Wildman-Crippen logP used in the estimate")
    ph: float = Field(description="Target pH used for the estimate")
    fraction_neutral: float = Field(description="Estimated fraction of the neutral (unionized) species at target pH")

    most_basic_pka: float | None = Field(description="Max predicted base-site pKa (None if no base sites predicted)")
    most_acidic_pka: float | None = Field(description="Min predicted acid-site pKa (None if no acid sites predicted)")
    num_basic_sites: int = Field(description="Number of predicted base sites")
    num_acidic_sites: int = Field(description="Number of predicted acid sites")

    mapped_smiles: str = Field(description="Atom-mapped SMILES from the pKa predictor")
    warnings: list[str] = Field(default_factory=list, description="Heuristic caveats and notes about this estimate")


################################################################################
# Public API
################################################################################


def compute_similarity(
    smiles: str,
    reference_smiles: list[str],
    fingerprint: FingerprintType = FingerprintType.MORGAN,
    standardize: bool = False,
) -> SimilarityOutput:
    """
    Compute fingerprint similarity between a query SMILES and a list of reference SMILES.

    Defaults are fixed: Morgan radius=2, 2048 bits, and chirality included.

    Args:
        smiles (str): Query SMILES.
        reference_smiles (list[str]): Reference SMILES to compare against.
        fingerprint (FingerprintType): Fingerprint type.
        standardize (bool): Standardize all SMILES first (remove salts, canonical tautomer).

    Returns:
        SimilarityOutput: Pydantic model with fields:
            - fingerprint (str): Fingerprint type used for comparison.
            - similarities (list[SimilarityEntry]): Sorted similarity results.
              Each entry includes reference_smiles (str) and similarity (float, 0-1).
    """
    mol = _validate_smiles(smiles, standardize=standardize)
    refs = [_validate_smiles(ref, standardize=standardize) for ref in reference_smiles]

    fp_key = _coerce_enum(fingerprint, FingerprintType, InvalidFingerprintError)

    def get_fp(m: Chem.Mol) -> DataStructs.ExplicitBitVect:
        return _FINGERPRINT_BUILDERS[fp_key](m)

    qfp = get_fp(mol)
    ref_fps = [get_fp(m) for m in refs]

    # C++ bulk compute (much faster than Python loop)
    sims = DataStructs.BulkTanimotoSimilarity(qfp, ref_fps)

    similarities = [
        SimilarityEntry(reference_smiles=ref_smi, similarity=_round_output(float(sim)))
        for ref_smi, sim in zip(reference_smiles, sims, strict=False)
    ]
    similarities.sort(key=lambda x: x.similarity, reverse=True)

    return SimilarityOutput(fingerprint=fp_key.value, similarities=similarities)


def find_mcs(
    smiles: str,
    reference_smiles: list[str],
    complete_rings_only: bool = True,
    ring_matches_ring_only: bool = True,
    standardize: bool = False,
) -> MCSOutput:
    """
    Find the maximum common substructure (MCS) across query + reference SMILES.

    Args:
        smiles (str): Query SMILES.
        reference_smiles (list[str]): Reference SMILES to include in the MCS search.
        complete_rings_only (bool): MCS must contain complete rings, not partial (default True).
        ring_matches_ring_only (bool): Ring atoms only match other ring atoms (default True).
        standardize (bool): Standardize all SMILES first (remove salts, canonical tautomer).

    Returns:
        MCSOutput: Pydantic model with fields:
            - smarts (str): SMARTS pattern of the MCS.
            - smiles (str | None): SMILES form of the MCS (if convertible).
            - num_atoms (int): Number of atoms in the MCS.
            - num_bonds (int): Number of bonds in the MCS.
            - canceled (bool): Whether the search hit the timeout.
            - query_coverage (float): Fraction of query heavy atoms covered (0-1).
            - ref_coverages (list[float]): Fraction of each reference covered (0-1).
    """
    mol = _validate_smiles(smiles, standardize=standardize)
    refs = [_validate_smiles(ref, standardize=standardize) for ref in reference_smiles]

    mcs = rdFMCS.FindMCS(
        [mol, *refs],
        timeout=5,
        completeRingsOnly=complete_rings_only,
        ringMatchesRingOnly=ring_matches_ring_only,
    )

    mcs_smiles = None
    if mcs.smartsString:
        mcs_mol = Chem.MolFromSmarts(mcs.smartsString)
        if mcs_mol is not None:
            mcs_smiles = Chem.MolToSmiles(mcs_mol, canonical=True)

    mcs_atoms = int(mcs.numAtoms)
    query_atoms = mol.GetNumHeavyAtoms()
    query_coverage = mcs_atoms / query_atoms if query_atoms > 0 else 0.0
    ref_coverages = [mcs_atoms / r.GetNumHeavyAtoms() if r.GetNumHeavyAtoms() > 0 else 0.0 for r in refs]

    return MCSOutput(
        smarts=mcs.smartsString,
        smiles=mcs_smiles,
        num_atoms=mcs_atoms,
        num_bonds=int(mcs.numBonds),
        canceled=bool(mcs.canceled),
        query_coverage=_round_output(query_coverage),
        ref_coverages=_round_output(ref_coverages),
    )


def score_structural_alerts(
    smiles: str,
    alert_library: AlertLibrary = AlertLibrary.ALL,
    standardize: bool = False,
) -> StructuralAlertsOutput:
    """
    Screen a SMILES against RDKit's built-in structural alert catalogs.

    Args:
        smiles (str): Query SMILES.
        alert_library (AlertLibrary): Built-in RDKit library selector.
        standardize (bool): Standardize SMILES first (remove salts, canonical tautomer).

    Returns:
        StructuralAlertsOutput: Pydantic model with fields:
            - library (str): Alert library used.
            - count (int): Number of matched alerts.
            - alerts (list[AlertEntry]): Matches with details.
              Each entry includes description (str), filter_set (str | None),
              reference (str | None), and scope (str | None).
    """
    mol = _validate_smiles(smiles, standardize=standardize)
    key = _coerce_enum(alert_library, AlertLibrary, InvalidAlertLibraryError)

    params = FilterCatalogParams()
    params.AddCatalog(_ALERT_LIBRARY_MAP[key])
    fc = FilterCatalog(params)

    alerts = []
    for entry in fc.GetMatches(mol):
        props = {name: entry.GetProp(name) for name in entry.GetPropList()}
        alerts.append(
            AlertEntry(
                description=entry.GetDescription(),
                filter_set=props.get("FilterSet"),
                scope=props.get("Scope"),
            )
        )

    return StructuralAlertsOutput(library=key.value, count=len(alerts), alerts=alerts)


def extract_pharmacophore_features(smiles: str, standardize: bool = False) -> PharmacophoreOutput:
    """
    Extract pharmacophore-like features using RDKit's BaseFeatures definitions.

    Args:
        smiles (str): Query SMILES.
        standardize (bool): Standardize SMILES first (remove salts, canonical tautomer).

    Returns:
        PharmacophoreOutput: Pydantic model with fields:
            - feature_counts (dict[str, int]): Counts per feature family.
            - features (list[PharmacophoreFeature]): Detailed feature list.
              Each entry includes family (str), type (str), and atom_ids (list[int]).
    """
    mol = _validate_smiles(smiles, standardize=standardize)
    factory = _get_feature_factory()
    features = factory.GetFeaturesForMol(mol)

    counts = Counter()
    items = []
    for feat in features:
        family = feat.GetFamily()
        counts[family] += 1
        items.append(
            PharmacophoreFeature(
                family=family,
                type=feat.GetType(),
                atom_ids=list(feat.GetAtomIds()),
            )
        )

    return PharmacophoreOutput(feature_counts=dict(counts), features=items)


def classify_ionization(
    smiles: str,
    ph: float = 7.4,
    standardize: bool = False,
) -> IonizationOutput:
    """
    Classify the ionization state of a molecule at a target pH.

    Uses Dimorphite-DL to enumerate protonation states, then analyzes the distribution
    of charge states. Returns both summary statistics and a representative variant.

    Args:
        smiles (str): The SMILES string of the molecule.
        ph (float): Target pH for protonation (default: 7.4, physiological).
        standardize (bool): Standardize SMILES first (remove salts only, no tautomer canonicalization).

    Returns:
        IonizationOutput: Pydantic model with fields:
            - num_variants (int): Protonation variants considered.
            - net_charges (list[int]): Unique net charges observed.
            - charge_class_counts (dict[str, int]): Counts for acid/base/zwitterion/neutral.
            - has_positive_states (bool): Whether any variant is net positive.
            - has_negative_states (bool): Whether any variant is net negative.
            - is_ambiguous (bool): True when multiple net charges are present.
            - representative (IonizationRepresentative): Chosen variant with
              smiles (str), net_charge (int), and charge_class (str).
    """
    from dimorphite_dl import protonate_smiles

    # For ionization, only remove salts (no tautomer canonicalization)
    mol = _validate_smiles(smiles, standardize=standardize, canonical_tautomer=False)
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

    return IonizationOutput(
        num_variants=len(variant_data),
        net_charges=net_charges,
        charge_class_counts=dict(charge_class_counts),
        has_positive_states=any(c > 0 for c in net_charges),
        has_negative_states=any(c < 0 for c in net_charges),
        is_ambiguous=len(net_charges) > 1,
        representative=IonizationRepresentative(
            smiles=representative_data["smiles"],
            net_charge=representative_data["net_charge"],
            charge_class=representative_data["charge_class"],
        ),
    )


def standardize_smiles(
    smiles: str,
    remove_salts: bool = True,
    canonical_tautomer: bool = True,
    neutralize: bool = False,
) -> str:
    """
    Standardize a SMILES string with explicit control over each step.

    Args:
        smiles (str): The SMILES string to standardize.
        remove_salts (bool): Remove salts/counterions, keep largest organic fragment (default True).
        canonical_tautomer (bool): Canonicalize tautomers (default True).
        neutralize (bool): Neutralize formal charges (default False).
            WARNING: Setting True makes formal_charge descriptor meaningless.

    Returns:
        str: Canonical, standardized SMILES string (isomeric where available).
    """
    mol = _validate_smiles(
        smiles,
        standardize=remove_salts or canonical_tautomer,
        remove_salts=remove_salts,
        canonical_tautomer=canonical_tautomer,
    )

    if neutralize:
        try:
            uncharger = rdMolStandardize.Uncharger()
            mol = uncharger.uncharge(mol)
        except Exception as e:
            raise InvalidSMILESError(f"Error neutralizing SMILES string: {smiles}") from e

    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


# def get_functional_groups(smiles: str, standardize: bool = False) -> str:
#     """
#     Given a SMILES string, return the functional groups present in the molecule.

#     Args:
#         smiles (str): The SMILES string of the molecule.
#         standardize (bool): Standardize SMILES first (remove salts, canonical tautomer).

#     Returns:
#         str: Human-readable functional-group tree from AccFG (multi-line text).
#     """
#     mol = _validate_smiles(smiles, standardize=standardize)
#     smiles = Chem.MolToSmiles(mol, canonical=True)  # AccFG needs SMILES string

#     afg = AccFG()
#     fgs, fg_graph = afg.run(smiles, show_atoms=True, show_graph=True, canonical=True)

#     f = io.StringIO()
#     with redirect_stdout(f):
#         print_fg_tree(fg_graph, fgs.keys())

#     return f.getvalue()


def compute_descriptors(
    smiles: str,
    descriptors: list[DescriptorName | str] | None = None,
    standardize: bool = False,
) -> dict[str, float | int | str | bool | dict]:
    """
    Compute descriptors for a SMILES string.

    Args:
        smiles (str): The SMILES string of the molecule.
        descriptors (list[DescriptorName | str] | None): Optional list of descriptor names to compute. Defaults to all.
        standardize (bool): Standardize SMILES first (remove salts, canonical tautomer).

    Descriptors:
        Available descriptor names (pass via `descriptors`):
        - `masses`: molecular weights and formula
          - `average_mw`: average molecular weight (Da)
          - `formula`: molecular formula
        - `atom_counts`: atom counts
          - `heavy_atoms`: heavy atom count
          - `heteroatoms`: heteroatom count
          - `nitrogen`: nitrogen atom count
          - `oxygen`: oxygen atom count
          - `n_plus_o`: nitrogen + oxygen count
        - `surface_shape_props`: surface/shape descriptors
          - `tpsa`: topological polar surface area (A^2)
          - `molar_refractivity`: molar refractivity (Wildman-Crippen)
          - `fraction_csp3`: fraction of sp3 carbons (Lipinski Fsp3)
          - `stereocenters`: tetrahedral stereocenter count
        - `ring_counts`: ring descriptors
          - `total_rings`: total ring count
          - `aromatic_rings`: aromatic ring count
          - `saturated_rings`: saturated ring count
        - `logp`: octanol/water partition coefficient (Wildman-Crippen)
        - `num_rotatable_bonds`: rotatable bond count (Lipinski)
        - `num_amide_bonds`: amide bond count
        - `formal_charge`: net formal charge
        - `qed`: quantitative estimate of drug-likeness (0-1)
        - `hydrogen_bonding`: hydrogen bond donors/acceptors
          - `hbd`: hydrogen bond donor count
          - `hba`: hydrogen bond acceptor count
        - `lipinski_violations`: Lipinski Rule-of-5 thresholds
          - `mw_violation`: MW > 500
          - `logp_violation`: LogP > 5
          - `hbd_violation`: HBD > 5
          - `hba_violation`: HBA > 10
          - `num_violations`: total violations
        - `esol`: ESOL (Delaney) aqueous solubility estimate
          - `log_s_esol`: log10 mol/L
          - `solubility_mg_per_ml`: solubility in mg/mL
          - `solubility_class`: qualitative class label

    Returns:
        dict[str, float | int | str | bool | dict]: Mapping of descriptor names to values.
            Each top-level key corresponds to a requested descriptor section. Values are
            scalars (e.g., "logp": float) or nested dicts (e.g., "masses": {"average_mw": float, "formula": str}).
    """
    mol = _validate_smiles(smiles, standardize=standardize)

    requested = [_coerce_enum(d, DescriptorName, InvalidDescriptorError) for d in (descriptors or list(DescriptorName))]
    requested = list(dict.fromkeys(requested))
    return _round_output({name.value: _DESCRIPTOR_FNS[name](mol) for name in requested})


def match_substructure(
    smiles: str, patterns: dict[str, str], standardize: bool = False
) -> dict[str, SubstructureMatchResult]:
    """
    Test whether a molecule contains the given SMARTS substructures and count occurrences.

    Args:
        smiles (str): The SMILES string of the molecule. Do not pass in an ellipsis (`...`) or other abbreviation.
        patterns (dict[str, str]): Mapping of pattern name to SMARTS string.
        standardize (bool): Standardize SMILES first (remove salts, canonical tautomer).

    Returns:
        dict[str, SubstructureMatchResult]: Mapping from pattern name to result.
            Each result includes present (bool) and count (int).

    Raises:
        InvalidSMILESError: If *smiles* or any SMARTS pattern is invalid.
    """
    mol = _validate_smiles(smiles, standardize=standardize)
    results: dict[str, SubstructureMatchResult] = {}
    for name, smarts in patterns.items():
        query = Chem.MolFromSmarts(smarts)
        if query is None:
            raise InvalidSMILESError(f"Invalid SMARTS pattern for '{name}': {smarts}")
        matches = mol.GetSubstructMatches(query)
        results[name] = SubstructureMatchResult(present=len(matches) > 0, count=len(matches))
    return results


def analyze_ring_systems(smiles: str, standardize: bool = False) -> RingSystemsOutput:
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
        RingSystemsOutput: Pydantic model with fields:
            - ring_systems (list[RingSystemInfo]): Fused ring system details.
            - num_ring_systems (int): Total fused-system count.
            - largest_system_size (int): Max rings in any system.
            - largest_aromatic_system (int): Max aromatic rings in any system.
            - has_pah_like (bool): True if any system has >=3 fused aromatic rings.
            - largest_ring_size (int): Largest individual ring size.
            - has_macrocycle (bool): True if any ring has >=12 atoms.
            - num_macrocycles (int): Count of rings with >=12 atoms.
            - spiro_atoms (int): Number of spiro centers.
            - bridgehead_atoms (int): Number of bridgehead atoms.
    """
    mol = _validate_smiles(smiles, standardize=standardize)
    ring_info = mol.GetRingInfo()
    atom_rings: tuple[tuple[int, ...], ...] = ring_info.AtomRings()
    bond_rings: tuple[tuple[int, ...], ...] = ring_info.BondRings()

    if not atom_rings:
        return RingSystemsOutput(
            ring_systems=[],
            num_ring_systems=0,
            largest_system_size=0,
            largest_aromatic_system=0,
            has_pah_like=False,
            largest_ring_size=0,
            has_macrocycle=False,
            num_macrocycles=0,
            spiro_atoms=0,
            bridgehead_atoms=0,
        )

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
    systems: list[RingSystemInfo] = []
    largest_aromatic = 0
    largest_system = 0

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

        systems.append(
            RingSystemInfo(
                num_rings=num_rings,
                num_aromatic=aromatic_count,
                all_aromatic=aromatic_count == num_rings,
                atom_count=len(system_atoms),
                ring_sizes=sorted(ring_sizes),
                heteroatoms=sorted(heteroatom_symbols),
            )
        )

    systems.sort(key=lambda s: (-s.num_rings, -s.num_aromatic))

    # Macrocycle analysis (threshold: 12 atoms)
    all_ring_sizes = [len(ring) for ring in atom_rings]
    largest_ring_size = max(all_ring_sizes) if all_ring_sizes else 0
    num_macrocycles = sum(1 for size in all_ring_sizes if size >= 12)

    return RingSystemsOutput(
        ring_systems=systems,
        num_ring_systems=len(components),
        largest_system_size=largest_system,
        largest_aromatic_system=largest_aromatic,
        has_pah_like=largest_aromatic >= 3,
        largest_ring_size=largest_ring_size,
        has_macrocycle=num_macrocycles > 0,
        num_macrocycles=num_macrocycles,
        spiro_atoms=rdMolDescriptors.CalcNumSpiroAtoms(mol),
        bridgehead_atoms=rdMolDescriptors.CalcNumBridgeheadAtoms(mol),
    )


def get_murcko_scaffold(smiles: str, standardize: bool = False, generic: bool = False) -> MurckoScaffoldOutput:
    """
    Extract the Bemis-Murcko scaffold from a molecule.

    Args:
        smiles (str): The SMILES string of the molecule.
        standardize (bool): Standardize SMILES first (remove salts, canonical tautomer).
        generic (bool): Return generic scaffold (all atoms → carbon, all bonds → single).

    Returns:
        MurckoScaffoldOutput: Pydantic model with fields:
            - scaffold_smiles (str): Scaffold SMILES (empty if none).
            - generic_scaffold_smiles (str | None): Generic scaffold SMILES (if requested).
            - num_scaffold_atoms (int): Heavy atom count in scaffold.
            - num_scaffold_rings (int): Ring count in scaffold.
            - scaffold_fraction (float): Scaffold atoms / molecule heavy atoms.
    """
    mol = _validate_smiles(smiles, standardize=standardize)
    mol_heavy_atoms = mol.GetNumHeavyAtoms()

    core = None
    try:
        core = MurckoScaffold.GetScaffoldForMol(mol)
    except Exception:
        core = None

    has_core = core is not None and core.GetNumAtoms() > 0
    scaffold_smiles = Chem.MolToSmiles(core, canonical=True, isomericSmiles=True) if has_core else ""
    num_scaffold_atoms = core.GetNumHeavyAtoms() if has_core else 0  # ty:ignore[possibly-missing-attribute]
    num_scaffold_rings = core.GetRingInfo().NumRings() if has_core else 0  # ty:ignore[possibly-missing-attribute]

    generic_scaffold_smiles: str | None = None
    if generic:
        if has_core:
            generic_core = MurckoScaffold.MakeScaffoldGeneric(core)
            generic_scaffold_smiles = Chem.MolToSmiles(generic_core, canonical=True, isomericSmiles=False)
        else:
            generic_scaffold_smiles = ""

    return MurckoScaffoldOutput(
        scaffold_smiles=scaffold_smiles,
        generic_scaffold_smiles=generic_scaffold_smiles,
        num_scaffold_atoms=num_scaffold_atoms,
        num_scaffold_rings=num_scaffold_rings,
        scaffold_fraction=(_round_output(num_scaffold_atoms / mol_heavy_atoms) if mol_heavy_atoms > 0 else 0.0),
    )


_pka_predictor: Any = None


def predict_pka(
    smiles: str,
    standardize: bool = False,
) -> pKaPredictionOutput:
    """
    Predict pKa values for ionizable sites in a molecule.

    Args:
        smiles (str): The SMILES string of the molecule.
        standardize (bool): Standardize SMILES first (remove salts, canonical tautomer).

    Returns:
        pKaPredictionOutput: Pydantic model with fields:
            - base_sites (dict[int, float]): Base-site pKa values (1-indexed atom map numbers).
            - acid_sites (dict[int, float]): Acid-site pKa values (1-indexed atom map numbers).
            - most_basic_pka (float | None): Max base-site pKa (None if no base sites).
            - most_acidic_pka (float | None): Min acid-site pKa (None if no acid sites).
            - num_basic_sites (int): Number of base sites.
            - num_acidic_sites (int): Number of acid sites.
            - mapped_smiles (str): SMILES of the protonated molecule with atom map numbers set.
    """
    from molgpka import MolGpKa

    global _pka_predictor
    if _pka_predictor is None:
        _pka_predictor = MolGpKa(uncharged=True)

    mol = _validate_smiles(smiles, standardize=standardize)

    prediction = _pka_predictor.predict(mol)

    atom_smi = Chem.MolToSmiles(prediction.mol)
    base_sites = prediction.base_sites_1
    acid_sites = prediction.acid_sites_1

    return pKaPredictionOutput(
        base_sites=base_sites,
        acid_sites=acid_sites,
        most_basic_pka=(max(base_sites.values()) if base_sites else None),
        most_acidic_pka=(min(acid_sites.values()) if acid_sites else None),
        num_basic_sites=len(base_sites),
        num_acidic_sites=len(acid_sites),
        mapped_smiles=atom_smi,
    )


def estimate_logd(
    smiles: str,
    ph: float = 7.4,
    standardize: bool = False,
) -> LogDEstimateOutput:
    """
    Estimate logD at a target pH from predicted pKa values and RDKit logP.

    This tool uses a simple Henderson-Hasselbalch approximation to estimate the fraction
    of the neutral (unionized) species at the target pH, then computes:

        logD(pH) ≈ logP + log10(f_neutral)

    where logP is RDKit Wildman-Crippen logP. This is a heuristic intended for
    permeability/BBB-style reasoning and should not be treated as experimental logD.

    For polyprotic molecules, this uses a simple approximation based on the most basic
    and most acidic predicted pKa values (if present).

    Args:
        smiles (str): Query SMILES.
        ph (float): Target pH (default: 7.4).
        standardize (bool): Standardize SMILES first (remove salts, canonical tautomer).

    Returns:
        LogDEstimateOutput: Estimated logD plus supporting fields and warnings.
    """
    if not (0.0 <= ph <= 14.0):
        raise ModelRetry(f"pH must be between 0 and 14 (got {ph})")

    desc = compute_descriptors(smiles, descriptors=[DescriptorName.LOGP], standardize=standardize)
    logp = cast(float, desc["logp"])

    pka = predict_pka(smiles, standardize=standardize)

    warnings: list[str] = []
    if pka.num_basic_sites > 1:
        warnings.append("Multiple basic sites; using only most_basic_pka for neutral-fraction estimate.")
    if pka.num_acidic_sites > 1:
        warnings.append("Multiple acidic sites; using only most_acidic_pka for neutral-fraction estimate.")
    if pka.num_basic_sites > 0 and pka.num_acidic_sites > 0:
        warnings.append("Amphoteric molecule; neutral-fraction estimate assumes independent sites.")

    # Neutral fraction contributions:
    # - Bases: fraction unprotonated = 1 / (1 + 10^(pKa - pH))
    # - Acids: fraction protonated   = 1 / (1 + 10^(pH - pKa))
    f_neutral_base = 1.0
    if pka.most_basic_pka is not None:
        f_neutral_base = 1.0 / (1.0 + 10.0 ** (pka.most_basic_pka - ph))

    f_neutral_acid = 1.0
    if pka.most_acidic_pka is not None:
        f_neutral_acid = 1.0 / (1.0 + 10.0 ** (ph - pka.most_acidic_pka))

    f_neutral = f_neutral_base * f_neutral_acid
    if f_neutral <= 0.0:
        warnings.append("Estimated neutral fraction was non-positive; clamping for numerical stability.")
        f_neutral = 1e-12

    # Guard against log10(0) and absurd rounding artifacts.
    f_neutral = min(1.0, max(1e-12, f_neutral))
    if f_neutral < 1e-6:
        warnings.append("Estimated neutral fraction is extremely small; logD estimate may be unreliable.")

    logd = logp + math.log10(f_neutral)

    return LogDEstimateOutput(
        logd=_round_output(logd),
        logp=logp,
        ph=_round_output(ph),
        fraction_neutral=_round_output(f_neutral),
        most_basic_pka=pka.most_basic_pka,
        most_acidic_pka=pka.most_acidic_pka,
        num_basic_sites=pka.num_basic_sites,
        num_acidic_sites=pka.num_acidic_sites,
        mapped_smiles=pka.mapped_smiles,
        warnings=warnings,
    )


################################################################################
# Backend
################################################################################

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

_ALERT_LIBRARY_MAP: dict[AlertLibrary, FilterCatalogParams.FilterCatalogs] = {
    lib: getattr(FilterCatalogParams.FilterCatalogs, lib.name) for lib in AlertLibrary
}

_FEATURE_FACTORY = None


def _coerce_enum(value, enum_cls, error_cls):
    if isinstance(value, enum_cls):
        return value
    try:
        return enum_cls(value.lower().strip())
    except ValueError as exc:
        raise error_cls(f"Unknown '{value}'. Valid: {', '.join(v.value for v in enum_cls)}") from exc


def _get_feature_factory() -> "ChemicalFeatures.MolChemicalFeatureFactory":
    global _FEATURE_FACTORY
    if _FEATURE_FACTORY is None:
        fdef = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
        _FEATURE_FACTORY = ChemicalFeatures.BuildFeatureFactory(fdef)  # ty:ignore[unresolved-attribute]
    return _FEATURE_FACTORY


def _validate_smiles(
    smiles: str,
    standardize: bool = False,
    remove_salts: bool = True,
    canonical_tautomer: bool = True,
) -> Chem.Mol:
    """
    Validate a SMILES string and return the RDKit molecule object, optionally standardized.

    Args:
        smiles (str): The SMILES string to validate.
        standardize (bool): Apply standardization steps (default False).
        remove_salts (bool): If standardize=True, remove salts/counterions (default True).
        canonical_tautomer (bool): If standardize=True, canonicalize tautomers (default True).

    Returns:
        Chem.Mol: The RDKit molecule object.

    Raises:
        InvalidSMILESError: If the SMILES string is invalid.
    """
    if smiles.strip() == "":
        raise InvalidSMILESError(f"Invalid SMILES string: {smiles}")

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise InvalidSMILESError(f"Invalid SMILES string: {smiles}")

        if standardize:
            if remove_salts:
                lfc = rdMolStandardize.LargestFragmentChooser(preferOrganic=True)
                mol = lfc.choose(mol)
                if mol is None:
                    raise InvalidSMILESError("Standardization failed after fragment removal.")
            if canonical_tautomer:
                tautomer_enumerator = rdMolStandardize.TautomerEnumerator()
                mol = tautomer_enumerator.Canonicalize(mol)

        return mol
    except InvalidSMILESError:
        raise
    except Exception as e:
        raise InvalidSMILESError(f"Error parsing SMILES string: {smiles}") from e


def _compute_lipinski_violations(mol: Chem.Mol) -> dict[str, int | bool]:
    mw = Descriptors.MolWt(mol)  # ty:ignore[unresolved-attribute]
    logp = Descriptors.MolLogP(mol)  # ty:ignore[unresolved-attribute]
    hbd = Lipinski.NumHDonors(mol)  # ty:ignore[unresolved-attribute]
    hba = Lipinski.NumHAcceptors(mol)  # ty:ignore[unresolved-attribute]

    violations = {
        "mw_violation": mw > 500,
        "logp_violation": logp > 5,
        "hbd_violation": hbd > 5,
        "hba_violation": hba > 10,
    }

    violations["num_violations"] = sum(violations.values())

    return violations


def _compute_esol(mol: Chem.Mol) -> dict[str, float | str]:
    """ESOL (Delaney 2004, JCICS) aqueous solubility estimate."""
    logp = Descriptors.MolLogP(mol)  # ty:ignore[unresolved-attribute]
    mw = Descriptors.MolWt(mol)  # ty:ignore[unresolved-attribute]
    rb = Lipinski.NumRotatableBonds(mol)  # ty:ignore[unresolved-attribute]
    num_aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
    num_heavy = mol.GetNumHeavyAtoms()
    ap = num_aromatic_atoms / num_heavy if num_heavy > 0 else 0.0

    log_s = 0.16 - 0.63 * logp - 0.0062 * mw + 0.066 * rb - 0.74 * ap

    # 10^LogS mol/L * MW g/mol = g/L = mg/mL
    sol_mg_per_ml = (10**log_s) * mw

    if log_s >= -1:
        sol_class = "highly soluble"
    elif log_s >= -3:
        sol_class = "soluble"
    elif log_s >= -5:
        sol_class = "moderately soluble"
    else:
        sol_class = "poorly soluble"

    return dict(
        log_s_esol=round(log_s, 3),
        solubility_mg_per_ml=round(sol_mg_per_ml, 4),
        solubility_class=sol_class,
    )


def _round_output(value):
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, dict):
        return cast(T, {key: _round_output(item) for key, item in value.items()})
    if isinstance(value, list):
        return cast(T, [_round_output(item) for item in value])
    if isinstance(value, tuple):
        return cast(T, tuple(_round_output(item) for item in value))
    return value


_DESCRIPTOR_FNS: dict[DescriptorName, Callable[[Chem.Mol], float | int | str | bool | dict]] = {
    DescriptorName.MASSES: lambda mol: {
        "average_mw": Descriptors.MolWt(mol),  # ty:ignore[unresolved-attribute]
        "formula": rdMolDescriptors.CalcMolFormula(mol),
    },
    DescriptorName.ATOM_COUNTS: lambda mol: {
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "heteroatoms": rdMolDescriptors.CalcNumHeteroatoms(mol),
        "nitrogen": sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 7),
        "oxygen": sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 8),
        "n_plus_o": sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() in (7, 8)),
    },
    DescriptorName.SURFACE_SHAPE_PROPS: lambda mol: {
        "tpsa": Descriptors.TPSA(mol),  # ty:ignore[unresolved-attribute]
        "molar_refractivity": Descriptors.MolMR(mol),  # ty:ignore[unresolved-attribute]
        "fraction_csp3": Lipinski.FractionCSP3(mol),  # ty:ignore[unresolved-attribute]
        "stereocenters": rdMolDescriptors.CalcNumAtomStereoCenters(mol),
    },
    DescriptorName.RING_COUNTS: lambda mol: {
        "total_rings": Lipinski.RingCount(mol),  # ty:ignore[unresolved-attribute]
        "aromatic_rings": Lipinski.NumAromaticRings(mol),  # ty:ignore[unresolved-attribute]
        "saturated_rings": rdMolDescriptors.CalcNumSaturatedRings(mol),
    },
    DescriptorName.LOGP: lambda mol: Descriptors.MolLogP(mol),  # ty:ignore[unresolved-attribute]
    DescriptorName.NUM_AMIDE_BONDS: rdMolDescriptors.CalcNumAmideBonds,
    DescriptorName.NUM_ROTATABLE_BONDS: lambda mol: Lipinski.NumRotatableBonds(mol),  # ty:ignore[unresolved-attribute]
    DescriptorName.FORMAL_CHARGE: Chem.GetFormalCharge,
    DescriptorName.QED: QED.qed,
    DescriptorName.HYDROGEN_BONDING: lambda mol: {
        "hbd": Lipinski.NumHDonors(mol),  # ty:ignore[unresolved-attribute]
        "hba": Lipinski.NumHAcceptors(mol),  # ty:ignore[unresolved-attribute]
    },
    DescriptorName.LIPINSKI_VIOLATIONS: _compute_lipinski_violations,
    DescriptorName.ESOL: _compute_esol,
}

'''
This script is used to use AccFG to parse SMILES string into functional groups (or with attachment points).
'''

from rdkit import RDConfig
from rdkit.Chem import rdFMCS
import os, csv, re
from accfg import AccFG, draw_mol_with_fgs, molimg, compare_mols
from accfg.draw import print_fg_tree
from rdkit import Chem
from pathlib import Path
import json
from .rdkit_tools import _tool, _round_output
from pydantic import BaseModel, Field

Current_dir = Path(__file__).parent.resolve()

#-------------------------------------------
# Chached tools, precomputed FGs description
#-------------------------------------------

# Load FG cache
# We use the new precomputed JSONL file
FG_CACHE_PATH = Path(os.environ.get('NEMO_RL_FG_CACHE_PATH', str(Current_dir.parent / 'data' / 'TDC_all_fg_desc_with_attach_points_and_atom_ids.jsonl'))
FG_CACHE = {}
if FG_CACHE_PATH.exists():
    with open(FG_CACHE_PATH, 'r', encoding='utf-8') as f:
        # The file can be a single JSON object (legacy) or JSONL (new)
        # We try to load as JSON first, if it fails, we assume JSONL
        try:
            FG_CACHE = json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            for line in f:
                if line.strip():
                    try:
                        entry = json.loads(line)
                        FG_CACHE.update(entry)
                    except:
                        pass
else:
    print(f"Warning: FG cache not found at {FG_CACHE_PATH}")

from unittest.mock import patch
import multiprocessing


# Synchronous pool replacements to avoid daemonic subprocess issues
class SynchronousPool:
    def __init__(self, *args, **kwargs):
        pass
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    def map(self, func, iterable, chunksize=None):
        return list(map(func, iterable))
    def imap(self, func, iterable, chunksize=1):
        return map(func, iterable)
    def imap_unordered(self, func, iterable, chunksize=1):
        return map(func, iterable)
    def starmap(self, func, iterable, chunksize=None):
        return [func(*args) for args in iterable]
    def apply_async(self, func, args=(), kwds={}, callback=None):
        res = func(*args, **kwds)
        if callback: callback(res)
        class AsyncResult:
             def __init__(self, val): self.val = val
             def get(self, timeout=None): return self.val
             def successful(self): return True
             def wait(self, timeout=None): pass
        return AsyncResult(res)
    def close(self): pass
    def join(self): pass
    def terminate(self): pass

class SynchronousProcessPoolExecutor:
    def __init__(self, *args, **kwargs):
        # 可加日志确认是否命中
        # print("[SyncExecutor] ProcessPoolExecutor patched -> running synchronously")
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False  # 不吞异常

    def submit(self, fn, *args, **kwargs):
        # 直接同步执行
        value = fn(*args, **kwargs)

        class _FakeFuture:
            def __init__(self, v):
                self._v = v
            def result(self, timeout=None):
                return self._v

        return _FakeFuture(value)

    # 有些代码可能会用到 shutdown
    def shutdown(self, wait=True, cancel_futures=False):
        pass


def cached_describe_high_level_fg_fragments(smiles: str):
    '''
    Cache the result of high level FG fragments description, refer to shared_data/precalculate_fgs.py for more details.
    缓存高级 FG 片段描述的结果，有关更多详细信息，请参阅 shared_data/precalculate_fgs.py。
    '''
    if smiles in FG_CACHE:
        return FG_CACHE[smiles]

    # If not cached, compute and cache
    print(f"{smiles} not cached, computing...")
    # Prevent nested daemon pool error by forcing synchronous execution if the underlying library tries to spawn a pool
    with patch('multiprocessing.Pool', side_effect=SynchronousPool):
        return high_level_fg_fragments_w_attach_points_no_special_tokens_w_atom_ids(smiles)

#---------------------------------------------------
# Instant FG fragment tools, many different versions
#---------------------------------------------------

def fg_fragment_with_attachment_points(smiles: str, fg_atoms: tuple[int, ...]):
    mol = Chem.MolFromSmiles(smiles)
    fg = set(fg_atoms)

    # 1) 找到所有“跨边界”的 bond（一个端点在 fg，另一个不在）
    cut_bond_ids = []
    cut_bonds = []
    for b in mol.GetBonds():
        a1, a2 = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if (a1 in fg) ^ (a2 in fg):
            cut_bond_ids.append(b.GetIdx())
            cut_bonds.append((a1, a2))

    # 2) 断键并在断点加 dummy 原子（*）
    # FragmentOnBonds 会返回“一个整体 mol”，里面包含所有碎片，并用 dummy 标记断点  [oai_citation:3‡rdkit.readthedocs.io](https://rdkit.readthedocs.io/en/latest/GettingStartedInPython.html)

    if not cut_bond_ids:
        print(f"Warning: No cut bonds found for {smiles}, fg_atoms: {fg_atoms}")
        return None, None, None
    frag_mol = Chem.FragmentOnBonds(mol, cut_bond_ids, addDummies=True)

    # 3) 把整体 mol 拆成各个连通分量
    # GetMolFrags(asMols=False) 会给每个碎片的 atom id 列表；asMols=True 则直接返回 Mol 对象  [oai_citation:4‡buildmedia.readthedocs.org](https://buildmedia.readthedocs.org/media/pdf/rdkit/latest/rdkit.pdf?utm_source=chatgpt.com)
    frags_atom_ids = Chem.GetMolFrags(frag_mol, asMols=False, sanitizeFrags=False)

    # 4) 选出“和 fg 重叠最多”的那个碎片（通常就是你的功能团那块 + dummy）
    best_ids = max(frags_atom_ids, key=lambda ids: len(fg.intersection(ids)))

    # 5) 输出该碎片的 SMILES（会包含 dummy 连接点）
    frag_smiles = Chem.MolFragmentToSmiles(
        frag_mol,
        atomsToUse=list(best_ids),
        isomericSmiles=True
    )

    # （可选）也可以看看“断开后的整分子”长啥样，通常是多个组分用 '.' 分开
    all_fragged_smiles = Chem.MolToSmiles(frag_mol, isomericSmiles=True)

    return frag_smiles, all_fragged_smiles, cut_bonds


def describe_high_level_fg_fragments_with_attachment_points(smiles:str):
    '''
    Parse SMILES string into functional groups with attachment points
    Args:
        smiles (str): SMILES of the molecule
    Returns:
        FGs_description (str): description of founctional groups in the molecule with fragment SMILES and attachment points
    '''
    afg = AccFG(print_load_info=False)

    # show_atoms=True  -> 让输出带上“功能团命中的原子编号”
    # show_graph=True  -> 返回 fg_graph（用于打印树/结构化组织）
    fgs, fg_graph = afg.run(smiles, show_atoms=True, show_graph=True)

    FG_name_SMILES_fragment_map = {}
    
    for FG_name, FG_atom_ids_list in fgs.items():
        FG_fragment_smiles_list = []
        for FG_atom_ids in FG_atom_ids_list:
            FG_fragment_smiles, _, _ = fg_fragment_with_attachment_points(smiles, FG_atom_ids)
            if FG_fragment_smiles is None:
                FG_fragment_smiles_list.append('Not matched')
                continue
            FG_fragment_smiles_list.append(FG_fragment_smiles)
        
        FG_name_SMILES_fragment_map[FG_name] = FG_fragment_smiles_list

    FGs_description = f"The functional groups inside <SMILES>{smiles}</SMILES> are:\n"
    for i, (FG_name, FG_fragment_smiles_list) in enumerate(FG_name_SMILES_fragment_map.items()):
        FGs_description += f"{i+1}. {FG_name}:"
        FGs_description += f"\n   Count:{len(FG_fragment_smiles_list)}"
        FGs_description += "\n   Corresponding fragment SMILES:"
        for FG_fragment_smiles in FG_fragment_smiles_list:
            if FG_fragment_smiles == 'Not matched':
                FGs_description += " Not matched, "
                continue
            FGs_description += f" <SMILES>{FG_fragment_smiles}</SMILES>, "
        FGs_description += "\n"
    return FGs_description

#---------------------------------------------------
# Current most advanced version
#---------------------------------------------------

def high_level_fg_fragments_w_attach_points_no_special_tokens_w_atom_ids(smiles:str):
    '''
    Parse SMILES string into functional groups with attachment points and mark atom ids in the SMILES string for better structure description.
    Args:
        smiles (str): SMILES of the molecule
    Returns:
        FGs_description (str): SMILES string with atom ids and description of founctional groups in the molecule with fragment SMILES and attachment points
    '''
    afg = AccFG(print_load_info=False)
    mol_w_ids = Chem.MolFromSmiles(smiles)
    for atom in mol_w_ids.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    mol_wo_ids = Chem.MolFromSmiles(smiles)

    # show_atoms=True  -> 让输出带上“功能团命中的原子编号”
    # show_graph=True  -> 返回 fg_graph（用于打印树/结构化组织）
    fgs, fg_graph = afg.run(smiles, show_atoms=True, show_graph=True)

    # mark atom ids in smiles
    smiles_with_atom_ids = Chem.MolToSmiles(mol_w_ids, canonical=False)

    FG_name_SMILES_fragment_map = {}
    
    for FG_name, FG_atom_ids_list in fgs.items():
        FG_fragment_smiles_list = []
        for FG_atom_ids in FG_atom_ids_list:
            FG_fragment_smiles_attach, _, _ = fg_fragment_with_attachment_points(smiles, FG_atom_ids)
            if FG_fragment_smiles_attach is None:
                FG_fragment_smiles_list.append('No attachment point; FG is the entire molecule')
                continue
            
            FG_fragment_smiles_pure = Chem.MolFragmentToSmiles(
                mol_wo_ids,
                atomsToUse=list(FG_atom_ids),
                isomericSmiles=True,
                canonical=False,
            )
            FG_fragment_smiles_w_ids = Chem.MolFragmentToSmiles(
                mol_w_ids,
                atomsToUse=list(FG_atom_ids),
                isomericSmiles=True,
                canonical=False,
            )
            FG_fragment_smiles_list.append(f"{FG_fragment_smiles_pure} <-> {FG_fragment_smiles_w_ids} <-> {FG_fragment_smiles_attach}")
        
        FG_name_SMILES_fragment_map[FG_name] = FG_fragment_smiles_list

    FGs_description = f"Original SMILES: {smiles}\n"
    FGs_description += f"with atom ids marked: {smiles_with_atom_ids}.\n"
    FGs_description += "The functional groups inside the molecule are:\n"
    for i, (FG_name, FG_fragment_smiles_list) in enumerate(FG_name_SMILES_fragment_map.items()):
        FGs_description += f"{i+1}. {FG_name}:"
        FGs_description += f"\n   Count:{len(FG_fragment_smiles_list)}"
        FGs_description += "\n   Corresponding fragment SMILES <-> with atom ids <-> with attachment points:"
        for FG_fragment_smiles in FG_fragment_smiles_list:
            if FG_fragment_smiles == 'No attachment point; FG is the entire molecule':
                FGs_description += " No attachment point; FG is the entire molecule, "
                continue
            FGs_description += f" {FG_fragment_smiles}, "
        FGs_description += "\n"
    # FGs_description += "Be careful there might be over generalization for the name of the functional groups, double-check the corresponding fragment SMILES to make decisions. "
    # FGs_description += "Match the functional groups' atom ids and attachment points atom ids with the original SMILES string to better understand the whole structure."
    return FGs_description

#---------------------------------------------------

def describe_high_level_fg_fragments(smiles:str):
    '''
    Parse SMILES string into functional groups
    Args:
        smiles (str): SMILES of the molecule
    Returns:
        FGs_description (str): description of founctional groups in the molecule with fragment SMILES
    '''
    afg = AccFG(print_load_info=False)
    mol = Chem.MolFromSmiles(smiles)

    # show_atoms=True  -> 让输出带上“功能团命中的原子编号”
    # show_graph=True  -> 返回 fg_graph（用于打印树/结构化组织）
    fgs, fg_graph = afg.run(smiles, show_atoms=True, show_graph=True)

    FG_name_SMILES_fragment_map = {}
    
    for FG_name, FG_atom_ids_list in fgs.items():
        FG_fragment_smiles_list = []
        for FG_atom_ids in FG_atom_ids_list:
            FG_fragment_smiles = Chem.MolFragmentToSmiles(
                mol,
                atomsToUse=list(FG_atom_ids),
                isomericSmiles=True,
                canonical=False,   # 不强制重排，通常更便于对照
            )
            FG_fragment_smiles_list.append(FG_fragment_smiles)
        
        FG_name_SMILES_fragment_map[FG_name] = FG_fragment_smiles_list

    FGs_description = f"The functional groups inside <SMILES>{smiles}</SMILES> are:\n"
    for i, (FG_name, FG_fragment_smiles_list) in enumerate(FG_name_SMILES_fragment_map.items()):
        FGs_description += f"{i+1}. {FG_name}:"
        FGs_description += f"\n   Count:{len(FG_fragment_smiles_list)}"
        FGs_description += "\n   Corresponding fragment SMILES:"
        for FG_fragment_smiles in FG_fragment_smiles_list:
            # if FG_fragment_smiles == 'Not matched':
            #     FGs_description += " Not matched, "
            #     continue
            FGs_description += f" <SMILES>{FG_fragment_smiles}</SMILES>, "
        FGs_description += "\n"
    return FGs_description

def describe_high_level_fg_fragments_no_special_token(smiles:str):
    '''
    Parse SMILES string into functional groups
    Args:
        smiles (str): SMILES of the molecule
    Returns:
        FGs_description (str): description of founctional groups in the molecule with fragment SMILES
    '''
    afg = AccFG(print_load_info=False)
    mol = Chem.MolFromSmiles(smiles)

    # show_atoms=True  -> 让输出带上“功能团命中的原子编号”
    # show_graph=True  -> 返回 fg_graph（用于打印树/结构化组织）
    fgs, fg_graph = afg.run(smiles, show_atoms=True, show_graph=True)

    FG_name_SMILES_fragment_map = {}
    
    for FG_name, FG_atom_ids_list in fgs.items():
        FG_fragment_smiles_list = []
        for FG_atom_ids in FG_atom_ids_list:
            FG_fragment_smiles = Chem.MolFragmentToSmiles(
                mol,
                atomsToUse=list(FG_atom_ids),
                isomericSmiles=True,
                canonical=False,   # 不强制重排，通常更便于对照
            )
            FG_fragment_smiles_list.append(FG_fragment_smiles)
        
        FG_name_SMILES_fragment_map[FG_name] = FG_fragment_smiles_list

    FGs_description = f"The functional groups inside {smiles} are:\n"
    for i, (FG_name, FG_fragment_smiles_list) in enumerate(FG_name_SMILES_fragment_map.items()):
        FGs_description += f"{i+1}. {FG_name}:"
        FGs_description += f"\n   Count:{len(FG_fragment_smiles_list)}"
        FGs_description += "\n   Corresponding fragment SMILES:"
        for FG_fragment_smiles in FG_fragment_smiles_list:
            # if FG_fragment_smiles == 'Not matched':
            #     FGs_description += " Not matched, "
            #     continue
            FGs_description += f" {FG_fragment_smiles}, "  # Difference here, no <SMILES> token
        FGs_description += "\n"
    return FGs_description

#-------------------------------------------
# FG related but not AccFG
#-------------------------------------------

def match_substructure(
    smiles: str, patterns: dict[str, str], standardize: bool = False
) -> str:
    """
    Test whether a molecule contains the given SMARTS substructures and count occurrences.

    Args:
        smiles (str): The SMILES string of the molecule. Do not pass in an ellipsis (`...`) or other abbreviation.
        patterns (dict[str, str]): Mapping of pattern name to SMARTS string.
        standardize (bool): Standardize SMILES first (remove salts, canonical tautomer).

    Returns:
        str: A formatted string describing the substructure match results, which is easier for language models to read.

    Raises:
        ValueError: If *smiles* or any SMARTS pattern is invalid.
    """
    # mol = _validate_smiles(smiles, standardize=standardize)
    mol = Chem.MolFromSmiles(smiles)
    
    output = []
    for name, smarts in patterns.items():
        query = Chem.MolFromSmarts(smarts)
        if query is None:
            raise ValueError(f"Invalid SMARTS pattern for '{name}': {smarts}")
        matches = mol.GetSubstructMatches(query)
        count = len(matches)
        if count > 0:
            output.append(f"- {name}: Present (count: {count})")
        else:
            output.append(f"- {name}: Not present")
            
    if not output:
        return "No patterns provided for matching."
        
    return "Substructure Match Results:\n" + "\n".join(output)

#-------------------------------------------
# Structure related but not AccFG
#-------------------------------------------

def find_mcs(
    smiles: str,
    reference_smiles: list[str],
    complete_rings_only: bool = True,
    ring_matches_ring_only: bool = True,
    standardize: bool = False,
) -> str:
    """
    Find the maximum common substructure (MCS) across query + reference SMILES.

    Args:
        smiles (str): Query SMILES.
        reference_smiles (list[str]): Reference SMILES to include in the MCS search.
        complete_rings_only (bool): MCS must contain complete rings, not partial (default True).
        ring_matches_ring_only (bool): Ring atoms only match other ring atoms (default True).
        standardize (bool): Standardize all SMILES first (remove salts, canonical tautomer).

    Returns:
        str: A formatted string describing the MCS search results, which is easier for language models to read.
    """
    # mol = _validate_smiles(smiles, standardize=standardize)
    mol = Chem.MolFromSmiles(smiles)
    refs = [Chem.MolFromSmiles(ref) for ref in reference_smiles]

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

    output = (
        "Maximum Common Substructure (MCS) Results:\n"
        f"- smarts: {mcs.smartsString}\n"
        f"- smiles: {mcs_smiles}\n"
        f"- num_atoms: {mcs_atoms}\n"
        f"- num_bonds: {int(mcs.numBonds)}\n"
        f"- canceled: {bool(mcs.canceled)}\n"
        f"- query_coverage: {_round_output(query_coverage)}\n"
        f"- ref_coverages: {_round_output(ref_coverages)}"
    )

    return output

#-------------------------------------------
# AccFG OpenAI tool list
#-------------------------------------------

AccFG_OPENAI_TOOLS = [{
    'type': 'function',
    'function': {
        'name': 'describe_high_level_fg_fragments',
        'description': 'Accurately parse functional groups with attachment points, corresponding fragment SMILES, and atom ids in the SMILES string for better structure description.',
        'parameters': {
            'type': 'object',
            'properties': {
                'smiles': {
                    'type': 'string',
                    'description': 'The SMILES string to parse.'
                }
            },
            'required': [
                'smiles'
            ]
        }
    }
}, {
    'type': 'function',
    'function': {
        'name': 'match_substructure',
        'description': 'Test whether a molecule contains the given SMARTS substructures and count occurrences.',
        'parameters': {
            'type': 'object',
            'properties': {
                'smiles': {
                    'type': 'string',
                    'description': 'The SMILES string of the molecule. Do not pass in an ellipsis (`...`) or other abbreviation.'
                },
                'patterns': {
                    'type': 'object',
                    'description': 'Mapping of pattern name to SMARTS string.',
                    'additionalProperties': {
                        'type': 'string'
                    }
                }
            },
            'required': [
                'smiles',
                'patterns'
            ]
        }
    }
}, {
    'type': 'function',
    'function': {
        'name': 'find_mcs',
        'description': 'Find the maximum common substructure (MCS) across query + reference SMILES. The output is a single structure present in ALL provided molecules. To compare very different molecules, call this tool multiple times individually rather than placing all molecules in a single list.',
        'parameters': {
            'type': 'object',
            'properties': {
                'smiles': {
                    'type': 'string',
                    'description': 'Query SMILES.'
                },
                'reference_smiles': {
                    'type': 'array',
                    'items': {
                        'type': 'string'
                    },
                    'description': 'Reference SMILES to include in the MCS search.'
                }
            },
            'required': [
                'smiles',
                'reference_smiles'
            ]
        }
    }
}
]

if __name__ == "__main__":
    # example usage
    smiles = "CCC=CCC=CCC=CCCCCCCCC(=O)OS(C)(=O)=O"
    smiles_2 = "CCC=CCC=CCC=CCCC(=O)OS(C)(=O)=O"
    smiles_3 = "CCC=CCC=CCCCCCC=CCCCCCCCC(=O)OS(C)(=O)=O"
    # print(high_level_fg_fragments_w_attach_points_no_special_tokens_w_atom_ids(smiles))
    # print('\n')
    # print(describe_high_level_fg_fragments(smiles))
    # print('\n')
    # print(describe_high_level_fg_fragments_no_special_token(smiles))

    BBB_SMARTS_LIBRARY = {
        # Strong negatives / permanent charge
        "quaternary_ammonium": "[N+](C)(C)(C)C",
        "sulfonate": "S(=O)(=O)[O-]",

        # Acids that often create anionic species
        "carboxylic_acid": "C(=O)[O;H1,-1]",
        "tetrazole": "c1nnnn1",
        "phosphonate": "P(=O)(O)(O)",

        # Common basic centers
        "tertiary_amine": "[NX3;!$(NC=O);!$(NS(=O)=O)](C)(C)",
        "piperidine_like": "N1CCCCC1",
        "piperazine_like": "N1CCNCC1",

        # H-bond rich motifs
        "guanidine": "NC(=N)N",
        "urea": "NC(=O)N",
        "sulfonamide": "S(=O)(=O)N",
        "amide": "C(=O)N",

        # Highly polar substituents
        "nitro": "[N+](=O)[O-]",
        "poly_ether": "OCCO",  # crude proxy for PEG-like
        }

    print(match_substructure(smiles, BBB_SMARTS_LIBRARY))

    # print(find_mcs(smiles, [smiles_2, smiles_3]))

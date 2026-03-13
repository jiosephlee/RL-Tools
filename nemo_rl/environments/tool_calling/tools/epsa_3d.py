from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import AllChem
import math
import re
import tempfile
import os
from rdkit import RDConfig
from rdkit.Chem import ChemicalFeatures
from .rdkit_tools import _tool
import freesasa
freesasa.setVerbosity(freesasa.nowarnings)

def build_polar_atom_sets(mol_noH: Chem.Mol):
    """
    输入：不含显式H的分子（heavy-atom mol）
    输出：若干 set，元素是 heavy atom index（与 AddHs 后的 mol 的 heavy index 一致）
    """
    fdef = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
    factory = ChemicalFeatures.BuildFeatureFactory(fdef)
    feats = factory.GetFeaturesForMol(mol_noH)

    sets = {
        "Donor": set(),
        "Acceptor": set(),
        "PosIonizable": set(),
        "NegIonizable": set(),
        "CarbonylO": set(),
    }

    for ft in feats:
        fam = ft.GetFamily()
        if fam in sets:
            sets[fam].update(ft.GetAtomIds())

    # 额外：羰基氧（O= C）
    for a in mol_noH.GetAtoms():
        if a.GetSymbol() != "O":
            continue
        for b in a.GetBonds():
            if b.GetBondType() == Chem.BondType.DOUBLE and b.GetOtherAtom(a).GetSymbol() == "C":
                sets["CarbonylO"].add(a.GetIdx())
                break

    return sets

def rdkit_embed_minimize(smiles: str, n_confs: int = 50, seed: int = 0):
    """
    生成多个构象，并用 MMFF94s 最小化，返回：
    - mol: 含 H、含多个构象的 RDKit Mol
    - energies: [(confId, energy), ...] 按能量从低到高排序

    兜底策略：
    - 若 MMFF 参数失败，则 fallback 用 UFF 做最小化（但能量不可与 MMFF 混用比较）
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")

    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    params.numThreads = 0
    params.pruneRmsThresh = 0.25  # 去重：避免大量重复构象
    conf_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=int(n_confs), params=params))
    if not conf_ids:
        raise RuntimeError("Conformer embedding failed")

    # 尝试 MMFF
    props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
    energies = []

    if props is not None:
        for cid in conf_ids:
            ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=cid)
            if ff is None:
                continue
            ff.Minimize(maxIts=500)
            energies.append((cid, float(ff.CalcEnergy())))
    else:
        # fallback UFF
        for cid in conf_ids:
            ff = AllChem.UFFGetMoleculeForceField(mol, confId=cid)
            if ff is None:
                continue
            ff.Minimize(maxIts=500)
            energies.append((cid, float(ff.CalcEnergy())))

    if not energies:
        raise RuntimeError("Forcefield minimization failed for all conformers")

    energies.sort(key=lambda x: x[1])
    return mol, energies

def _get_atom_areas(result, n_atoms: int):
    # 新版：atomAreas() -> list
    if hasattr(result, "atomAreas"):
        return list(result.atomAreas())
    # 旧版：atomArea(i) -> float
    if hasattr(result, "atomArea"):
        return [float(result.atomArea(i)) for i in range(n_atoms)]
    raise AttributeError("FreeSASA Result has neither atomAreas() nor atomArea(i)")


def _parse_pdb_elements_in_order(pdb_block: str) -> list[str]:
    """
    兜底：从 PDB 文本中按 ATOM/HETATM 顺序解析元素符号。
    优先使用 element 字段(列 77-78)，没有则从 atom name 推断。
    """
    elems = []
    for line in pdb_block.splitlines():
        if not (line.startswith("ATOM  ") or line.startswith("HETATM")):
            continue
        elem = line[76:78].strip()
        if elem:
            elems.append(elem.capitalize())
            continue

        # fallback: 从 atom name 推断（处理 1CL / CL1 / Br 等情况）
        name = line[12:16].strip()
        name = name.lstrip("0123456789")
        m = re.match(r"^([A-Za-z]{1,2})", name)
        if not m:
            elems.append("")
        else:
            elems.append(m.group(1).capitalize())
    return elems


def exposed_polar_sasa_for_conf(
    mol: Chem.Mol,
    confId: int,
    probe_radius: float = 1.4,
    include_charged: bool = True,
    polar_mode: str = "features",  # "features" or "elements"
):
    """
    返回：
      polar_sasa: 按选择的 polar_mode 得到的“极性暴露面积”(Å^2)
      total_sasa: 总SASA(Å^2)（包含H的话就是含H总SASA）
      breakdown: 各类别的SASA分解（Å^2）
    """

    # ---------- freesasa 计算 atom_areas（你现有这段保持不变） ----------
    pdb = Chem.MolToPDBBlock(mol, confId=confId)

    fs_params = freesasa.Parameters()
    fs_params.setProbeRadius(float(probe_radius))

    fs_options = {"hetatm": True, "hydrogen": True, "skip-unknown": False, "halt-at-unknown": False}

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as f:
            f.write(pdb)
            tmp_path = f.name

        structure = freesasa.Structure(tmp_path, options=fs_options)
        result = freesasa.calc(structure, fs_params)

        n_fs = structure.nAtoms()
        # 兼容不同 freesasa 版本
        if hasattr(result, "atomAreas"):
            atom_areas = list(result.atomAreas())
        else:
            atom_areas = [float(result.atomArea(i)) for i in range(n_fs)]
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    n_rd = mol.GetNumAtoms()
    if n_fs != n_rd:
        raise RuntimeError(f"Atom count mismatch: freesasa nAtoms={n_fs}, RDKit nAtoms={n_rd}")

    total_sasa = float(sum(atom_areas))

    # ---------- 只对 heavy atoms 做“化学角色”分类（更合理） ----------
    mol_noH = Chem.RemoveHs(mol)
    n_heavy = mol_noH.GetNumAtoms()

    # sanity check：确保 heavy atoms 在 AddHs 后仍是前 n_heavy 个 index
    for i in range(n_heavy):
        if mol.GetAtomWithIdx(i).GetSymbol() != mol_noH.GetAtomWithIdx(i).GetSymbol():
            raise RuntimeError("Heavy-atom index mapping unexpected; cannot safely map features to SASA.")

    breakdown = {}

    # 总SASA（重原子 vs 全原子）
    heavy_total = float(sum(atom_areas[:n_heavy]))
    breakdown["total_sasa_all_atoms"] = total_sasa
    breakdown["total_sasa_heavy_atoms"] = heavy_total

    # --------- 模式1：用 ChemicalFeatures（Donor/Acceptor/Ionizable）定义极性 ---------
    if polar_mode == "features":
        sets = build_polar_atom_sets(mol_noH)

        # 各类分解（注意：Donor/Acceptor 可能重叠，下面是“各自面积”，不是互斥分解）
        breakdown["donor_sasa"] = float(sum(atom_areas[i] for i in sets["Donor"]))
        breakdown["acceptor_sasa"] = float(sum(atom_areas[i] for i in sets["Acceptor"]))
        breakdown["pos_ionizable_sasa"] = float(sum(atom_areas[i] for i in sets["PosIonizable"]))
        breakdown["neg_ionizable_sasa"] = float(sum(atom_areas[i] for i in sets["NegIonizable"]))
        breakdown["carbonylO_sasa"] = float(sum(atom_areas[i] for i in sets["CarbonylO"]))

        # 定义“极性原子集合”：Donor ∪ Acceptor ∪ Ionizable（再可选加 formal charge）
        polar_idx = set().union(
            sets["Donor"], sets["Acceptor"], sets["PosIonizable"], sets["NegIonizable"]
        )

        if include_charged:
            for i in range(n_heavy):
                if mol_noH.GetAtomWithIdx(i).GetFormalCharge() != 0:
                    polar_idx.add(i)

        polar_sasa = float(sum(atom_areas[i] for i in polar_idx))
        breakdown["polar_idx_count"] = len(polar_idx)
        breakdown["polar_sasa_heavy_atoms"] = polar_sasa
        breakdown["polar_fraction_heavy"] = polar_sasa / heavy_total if heavy_total > 0 else float("nan")

        # ===================== 新增：all-atom 自洽版本（把 polar heavy 上的 H 也算为 polar） =====================
        polar_idx_all = set(polar_idx)

        # 把“连接到极性 heavy atom”的 H 也加入极性集合
        for hi in range(n_heavy, n_rd):  # AddHs 后，H 通常在 heavy atoms 之后
            atom = mol.GetAtomWithIdx(hi)
            if atom.GetSymbol() != "H":
                continue
            nbrs = atom.GetNeighbors()
            if not nbrs:
                continue
            heavy_nbr = nbrs[0]  # H 只连一个邻居
            if heavy_nbr.GetIdx() in polar_idx:
                polar_idx_all.add(hi)

        polar_sasa_all = float(sum(atom_areas[i] for i in polar_idx_all))
        polar_fraction_all_consistent = polar_sasa_all / total_sasa if total_sasa > 0 else float("nan")

        breakdown["polar_sasa_all_atoms"] = polar_sasa_all
        breakdown["polar_fraction_all_consistent"] = polar_fraction_all_consistent
        breakdown["polar_idx_all_count"] = len(polar_idx_all)

        # （可选）保留你原来的“非自洽 all”作为参考下界，输出“按全原子总面积”的比例（注意包含H会让分母变大）
        breakdown["polar_fraction_all_lowerbound"] = polar_sasa / total_sasa if total_sasa > 0 else float("nan")
        # ===================== 新增结束 =====================

        return polar_sasa_all, total_sasa, breakdown

    # --------- 模式2：退回元素集合（你原先那套） ---------
    elif polar_mode == "elements":
        polar_elems = {"O", "N", "S", "P"}
        polar_sasa = 0.0
        for i in range(n_heavy):
            atom = mol_noH.GetAtomWithIdx(i)
            if atom.GetSymbol() in polar_elems or (include_charged and atom.GetFormalCharge() != 0):
                polar_sasa += float(atom_areas[i])
        breakdown["polar_sasa_heavy_atoms"] = polar_sasa
        breakdown["polar_fraction_heavy"] = polar_sasa / heavy_total if heavy_total > 0 else float("nan")
        breakdown["polar_fraction_all"] = polar_sasa / total_sasa if total_sasa > 0 else float("nan")
        return polar_sasa, total_sasa, breakdown

    else:
        raise ValueError("polar_mode must be 'features' or 'elements'")



def exposed_polar_sasa_ensemble(
    smiles: str,
    n_confs: int = 80,
    top_k: int = 10,
    probe_radius: float = 1.4,
    boltzmann_T: float | None = 298.15,
):
    """
    生成构象 -> 最小化 -> 取 top_k 低能构象 -> 计算每个构象的 ePSA/totalSASA/polar_fraction
    可选：返回 Boltzmann 加权平均（仅当使用同一力场能量、且能量单位可比时更合理）
    """
    mol, energies = rdkit_embed_minimize(smiles, n_confs=n_confs)
    top = energies[: max(1, int(top_k))]

    rows = []
    for cid, e in top:
        polar, total, info = exposed_polar_sasa_for_conf(mol, cid, probe_radius=probe_radius, polar_mode="features")
        rows.append(
            {
                "confId": cid,
                "E": e,
                "polar_sasa": polar,                 # ✅加上
                "total_sasa": total,                 # ✅加上
                "polar_fraction": info["polar_fraction_all_consistent"],  # ✅给一个统一字段，后面统计用 polar_fraction_all 也可以
                **info
            }
        )

    # 统计：min/median/mean
    polar_vals = [r["polar_sasa"] for r in rows]
    frac_vals = [r["polar_fraction"] for r in rows]

    stats = {
        "polar_sasa_min": min(polar_vals),
        "polar_sasa_mean": sum(polar_vals) / len(polar_vals),
        "polar_fraction_min": min(frac_vals),
        "polar_fraction_mean": sum(frac_vals) / len(frac_vals),
    }

    # Boltzmann 加权（可选）
    boltz = None
    if boltzmann_T is not None and len(rows) >= 2:
        # RDKit MMFF/UFF 的能量通常可以当作 kcal/mol 量级做相对权重（近似）
        # 权重 w_i = exp(-(E_i - Emin)/(R*T)), R=0.001987 kcal/mol/K
        R = 0.0019872041
        Emin = rows[0]["E"]
        ws = [math.exp(-(r["E"] - Emin) / (R * boltzmann_T)) for r in rows]
        Z = sum(ws)
        ws = [w / Z for w in ws]
        boltz = {
            "polar_sasa_boltz": sum(w * r["polar_sasa"] for w, r in zip(ws, rows)),
            "polar_fraction_boltz": sum(w * r["polar_fraction"] for w, r in zip(ws, rows)),
        }

    return rows, stats, boltz


def get_3d_exposed_polar_surface(smiles: str):
    """
    Tool wrapper around exposed_polar_sasa_ensemble.

    Success return (dict):
      {
        "polar_sasa": float,
        "polar_fraction": float,
        "aggregation": "boltzmann" | "mean"
      }

    Failure return (str):
      "ERROR: <ExceptionType>: <message> | smiles=... | params=..."
    """
    try:
        # --- basic guardrails (optional but helps agents avoid silly calls) ---
        if not isinstance(smiles, str) or not smiles.strip():
            return "ERROR: ValueError: smiles must be a non-empty string"

        rows, stats, boltz = exposed_polar_sasa_ensemble(
            smiles=smiles,
        )

        # ✅ Boltz 可用：只返回 boltz
        if (
            boltz is not None
            and "polar_sasa_boltz" in boltz
            and "polar_fraction_boltz" in boltz
            and boltz["polar_sasa_boltz"] is not None
            and boltz["polar_fraction_boltz"] is not None
        ):
            return (
                f"3D conformation based estimation of PSA: {boltz['polar_sasa_boltz']:.2f}\n"
                f"3D conformation based estimation of Polar Fraction: {boltz['polar_fraction_boltz']:.2f}"
            )

        # ✅ Boltz 不可用：只返回 mean（stats）
        return (
            f"3D conformation based estimation of PSA: {stats['polar_sasa_mean']:.2f}\n"
            f"3D conformation based estimation of Polar Fraction: {stats['polar_fraction_mean']:.2f}"
        )

    except Exception as e:
        # 失败时按你的要求：返回一个字符串给模型看（不抛异常）
        et = type(e).__name__
        msg = str(e).strip() or "<no message>"
        return (
            f"ERROR: {et}: {msg}"
        )

SASA_OPENAI_TOOLS = [_tool('get_3d_exposed_polar_surface', 'Return the 3D conformation based estimation of exposed polar surface area and its fraction of the total 3D solvent-accessible surface area.')]

if __name__ == "__main__":
    smiles = "C[C@]12C[C@H]([C@@H]([C@@]1(CC(=O)[C@@]3([C@H]2CC=C4[C@H]3C=C(C(=O)C4(C)C)O)C)C)[C@](C)(C(=O)/C=C\\C(C)(C)O)O)O"  # 你的 SMILES
    rows, stats, boltz = exposed_polar_sasa_ensemble(
        smiles,
        n_confs=120,
        top_k=15,
        probe_radius=1.4,
        boltzmann_T=298.15,
    )

    for r in rows:
        print(r)
    print("stats:", stats)
    print("boltz:", boltz)

    print(get_3d_exposed_polar_surface(smiles))

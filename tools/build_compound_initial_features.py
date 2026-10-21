"""
HyperCTI 项目 - 构建化合物初始特征
────────────────────────────────────────────────────────────────────────────
输入：compound_smiles_pubchem.txt（由 fetch_smiles_from_pubchem.py 生成）
输出（保存到 dataset/TCMsuite/features/newfeatures/）：
  c_smiles_tokens.npy   shape: (N, max_seq_len)
  c_fingerprints.npy    shape: (N, 2513)
  c_graph_feat.npy      shape: (N, max_atoms, 133)
  c_graph_adj.npy       shape: (N, max_atoms, max_atoms)
  clist_full.txt / clist_full
  compound_smiles.csv
  invalid_smiles.txt    （存在无效 SMILES 时生成）

目录结构：
  HyperCTI-main4/                             ← repo_root（项目根）
    ├── MVML-MPI-main/MVML-MPI-main/
    │     └── pubchemfp.py
    └── dataset/TCMsuite/features/
          ├── newfeatures/                 ← 输出目录
          └── tools/
                ├── build_compound_initial_features.py  ← 本脚本
                └── output/
                      └── compound_smiles_pubchem.txt   ← 输入

依赖：pip install rdkit numpy
用法：
  # 使用默认路径
  python build_compound_initial_features.py

  # 自定义路径
  python build_compound_initial_features.py \\
      --input   dataset/TCMsuite/features/tools/output/compound_smiles_pubchem.txt \\
      --output_dir dataset/TCMsuite/features/newfeatures \\
      --max_seq_len 250 --max_atoms 120
"""

import argparse
import importlib.util
import os
from pathlib import Path
import pandas as pd

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

# ─── Token 表 ─────────────────────────────────────────────────────────────────
# 在原有基础上补充 cache SMILES 中出现的字符：
#   % → 环编号 ≥10（如 %10, %11）
#   0 9 → 数字环编号
#   a   → 部分芳香原子写法
TOKENS = [
    " ", "<unk>", "C", "O", "(", ")", "c", "=", "1", "2", "N", "3", "n", "P", "4", "[", "]", "S",
    "H", "5", "l", "-", "*", "o", "+", "6", "#", "M", "F", "g", "7", "B", "r", "s", "I", "e", "i",
    "8", "Z",
    # ↓ 新增字符
    "%", "0", "9", "a", "T",    
    # ↓ 新增字符 - 20260724
    'u', 'm', '.', 'R', 'K', 'L', 'Y', 'W', 'A', 'b'
    # isomeric symbol
    # 'm', 'W', '/', '@', 'A', 'Y', 'k', 'u', 'p', 'b', 'R', 'h', 'L', 'U', 'K', ':', 'f', 'E', '\\', '.', 't', 
    
]
TOKEN_TO_ID = {token: idx for idx, token in enumerate(TOKENS)}
PAD_ID = TOKEN_TO_ID[" "]
UNK_ID = TOKEN_TO_ID["<unk>"]

# ─── 原子特征定义 ─────────────────────────────────────────────────────────────
ATOM_TYPE_MAX = 100
ATOM_FEATURES_DEFINE = {
    "atom_symbol": list(range(ATOM_TYPE_MAX)),
    "degree": [0, 1, 2, 3, 4, 5],
    "formal_charge": [-1, -2, 1, 2, 0],
    "charity_type": [0, 1, 2, 3],
    "hydrogen": [0, 1, 2, 3, 4],
    "hybridization": [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ],
}

# ─── 模块加载 ─────────────────────────────────────────────────────────────────
def load_pubchem_module(repo_root: Path):
    pubchem_path = repo_root / "MVML-MPI-main" / "pubchemfp.py"
    spec = importlib.util.spec_from_file_location("pubchemfp_local", str(pubchem_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 PubChem 指纹模块: {pubchem_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "GetPubChemFPs"):
        raise RuntimeError(f"模块缺少 GetPubChemFPs: {pubchem_path}")
    return module.GetPubChemFPs

# ─── 文件解析 ─────────────────────────────────────────────────────────────────

def parse_smiles_file(path: Path):
    compound_ids, smiles_list = [], []
    smiles_df = pd.read_csv(str(path))
    for _, row in smiles_df.iterrows():
        compound_id = row["compound_id"]
        smiles = row["SMILES"]
        compound_ids.append(str(compound_id).strip())
        smiles_list.append(smiles.strip())
    return compound_ids, smiles_list

# ─── 特征计算函数 ─────────────────────────────────────────────────────────────

def encode_smiles(smiles: str, max_seq_len: int) -> np.ndarray:
    vec = np.full((max_seq_len,), PAD_ID, dtype=np.int32)
    for i, ch in enumerate(smiles[:max_seq_len]):
        vec[i] = TOKEN_TO_ID.get(ch, UNK_ID)
    return vec


def onek_encoding_unk(key, candidates):
    encoding = [0] * (len(candidates) + 1)
    index = candidates.index(key) if key in candidates else -1
    encoding[index] = 1
    return encoding


def get_atom_feature(atom) -> np.ndarray:
    feature = (
        onek_encoding_unk(atom.GetAtomicNum() - 1, ATOM_FEATURES_DEFINE["atom_symbol"])
        + onek_encoding_unk(atom.GetTotalDegree(),  ATOM_FEATURES_DEFINE["degree"])
        + onek_encoding_unk(atom.GetFormalCharge(), ATOM_FEATURES_DEFINE["formal_charge"])
        + onek_encoding_unk(int(atom.GetChiralTag()), ATOM_FEATURES_DEFINE["charity_type"])
        + onek_encoding_unk(int(atom.GetTotalNumHs()), ATOM_FEATURES_DEFINE["hydrogen"])
        + onek_encoding_unk(int(atom.GetHybridization()), ATOM_FEATURES_DEFINE["hybridization"])
        + [1 if atom.GetIsAromatic() else 0]
        + [atom.GetMass() * 0.01]
    )
    return np.asarray(feature, dtype=np.float32)


def build_graph_features(mol: Chem.Mol, max_atoms: int):
    atom_feat_dim = 133
    feat = np.zeros((max_atoms, atom_feat_dim), dtype=np.float32)
    adj  = np.zeros((max_atoms, max_atoms),     dtype=np.float32)
    atom_count = min(mol.GetNumAtoms(), max_atoms)

    for i in range(atom_count):
        feat[i] = get_atom_feature(mol.GetAtomWithIdx(i))

    raw_adj = Chem.GetAdjacencyMatrix(mol).astype(np.float32)
    raw_adj = raw_adj[:atom_count, :atom_count]
    raw_adj = raw_adj + np.eye(atom_count, dtype=np.float32)
    degree  = np.sum(raw_adj, axis=1)
    inv_sqrt = np.power(degree, -0.5, where=degree > 0)
    inv_sqrt[degree <= 0] = 0.0
    norm_adj = inv_sqrt[:, None] * raw_adj * inv_sqrt[None, :]
    adj[:atom_count, :atom_count] = norm_adj
    return feat, adj


def bitvect_to_array(bitvect) -> np.ndarray:
    arr = np.zeros((bitvect.GetNumBits(),), dtype=np.int32)
    DataStructs.ConvertToNumpyArray(bitvect, arr)
    return arr


def build_fingerprint(mol: Chem.Mol, get_pubchem_fps) -> np.ndarray:
    fp_maccs   = bitvect_to_array(AllChem.GetMACCSKeysFingerprint(mol))
    fp_erg     = np.asarray(
        AllChem.GetErGFingerprint(mol, fuzzIncrement=0.3, maxPath=21, minPath=1),
        dtype=np.float32,
    )
    fp_erg     = (fp_erg > 0).astype(np.int32)
    fp_pubchem = np.asarray(get_pubchem_fps(mol), dtype=np.int32)
    fp_ecfp2   = bitvect_to_array(
        AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    )
    return np.concatenate([fp_maccs, fp_erg, fp_pubchem, fp_ecfp2]).astype(np.int32)

# ─── 主函数 ───────────────────────────────────────────────────────────────────

def main():
    RDLogger.DisableLog("rdApp.warning")
    RDLogger.DisableLog("rdApp.error")

    dataset_type = "TCMsuite"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        # 修改1：默认输入改为 pubchem 版本
        default=f"datasets/{dataset_type}/canonical_smiles.csv",
        help="SMILES 输入文件(id1\\tSMILES 格式）",
    )
    
    parser.add_argument(
        "--all_compounds",
        default=f"datasets/{dataset_type}/compound_ids.txt",
        help="记录所有compound id文件",
    )

    parser.add_argument(
        "--output_dir",
        default=f"datasets/{dataset_type}/features/newfeatures",
        help="特征输出目录",
    )
    parser.add_argument(
        "--max_seq_len", type=int, default=250,
        # 修改2：从 140 → 250，覆盖 cache 中 17 条超长 SMILES（最长 801，但那是极端大分子，截断可接受）
        # 若要完整保留所有分子序列，可设为 810，但会显著增加内存
        help="SMILES token 序列最大长度（截断超出部分）",
    )
    parser.add_argument(
        "--max_atoms", type=int, default=120,
        # 修改3：从 100 → 120，覆盖更多分子；极端大分子（>120原子）仍会截断图结构
        help="分子图最大原子数（超出部分截断）",
    )
    args = parser.parse_args()

    # 修改4：repo_root 从 parent.parent 改为 parent * 4
    # 脚本路径: HyperCTI-main4/dataset/TCMsuite/features/tools/build_compound_initial_features.py
    # parent(1)=tools, parent(2)=features, parent(3)=TCMsuite, parent(4)=dataset, parent(5)≠root
    # 实测：脚本在 tools/ 下，parent.parent = features/（原代码以此为 repo_root 是错的）
    # MVML-MPI-main 在 HyperCTI-main4/ 下，所以 repo_root = tools/ 的上4级
    repo_root = Path(__file__).resolve().parents[1]  # HyperCTI-main4/

    input_path = (
        (repo_root / args.input).resolve()
        if not os.path.isabs(args.input)
        else Path(args.input)
    )
    output_dir = (
        (repo_root / args.output_dir).resolve()
        if not os.path.isabs(args.output_dir)
        else Path(args.output_dir)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"repo_root  : {repo_root}")
    print(f"输入文件   : {input_path}")
    print(f"输出目录   : {output_dir}")
    print(f"max_seq_len: {args.max_seq_len}")
    print(f"max_atoms  : {args.max_atoms}")
    print()

    get_pubchem_fps = load_pubchem_module(repo_root)
    compound_ids, smiles_list = parse_smiles_file(input_path)
    
    with open(str(repo_root) + f"/{args.all_compounds}", "r") as f:
        compounds = f.read()
    compounds = compounds.split("\n")
    compounds = {c_id: idx for idx, c_id in enumerate(compounds)}
    n_compounds = len(compounds)
    if n_compounds == 0:
        raise RuntimeError(f"输入文件为空或无法解析: {input_path}")


    smiles_tokens = np.zeros((n_compounds, args.max_seq_len), dtype=np.int32)
    fingerprints  = np.zeros((n_compounds, 2513),             dtype=np.int32)
    graph_feat    = np.zeros((n_compounds, args.max_atoms, 133),              dtype=np.float32)
    graph_adj     = np.zeros((n_compounds, args.max_atoms, args.max_atoms),   dtype=np.float32)

    invalid = []
    for _, (cid, smiles) in enumerate(zip(compound_ids, smiles_list)):
        i = compounds.get(str(cid), None)
        if i is None: continue
        smiles_tokens[i] = encode_smiles(smiles, args.max_seq_len)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            invalid.append(cid)
            continue
        fingerprints[i]          = build_fingerprint(mol, get_pubchem_fps)
        graph_feat[i], graph_adj[i] = build_graph_features(mol, args.max_atoms)

    # ── 保存特征 ──────────────────────────────────────────────────────────────
    np.save(output_dir / "c_smiles_tokens.npy", smiles_tokens)
    np.save(output_dir / "c_fingerprints.npy",  fingerprints)
    np.save(output_dir / "c_graph_feat.npy",    graph_feat)
    np.save(output_dir / "c_graph_adj.npy",     graph_adj)

    # ── 保存 ID 列表 ──────────────────────────────────────────────────────────
    # for fname in ("clist_full.txt", "clist_full"):
    #     with (output_dir / fname).open("w", encoding="utf-8") as fh:
    #         for cid in compound_ids:
    #             fh.write(f"{cid}\n")

    # ── 保存 SMILES CSV ───────────────────────────────────────────────────────
    invalid_set = set(invalid)
    with (output_dir / "compound_smiles.csv").open("w", encoding="utf-8") as fh:
        fh.write("compound_id,smiles,is_valid\n")
        for cid, smiles in zip(compound_ids, smiles_list):
            valid = 0 if cid in invalid_set else 1
            safe_smiles = smiles.replace('"', '""')
            fh.write(f'{cid},"{safe_smiles}",{valid}\n')

    # # ── 保存无效 SMILES 列表 ──────────────────────────────────────────────────
    if invalid:
        with (output_dir / "invalid_smiles.txt").open("w", encoding="utf-8") as fh:
            for cid in invalid:
                fh.write(f"{cid}\n")

    # ── 打印汇总 ──────────────────────────────────────────────────────────────
    print(f"化合物总数       : {n_compounds}")
    print(f"无效 SMILES 数量 : {len(invalid)}")
    print(f"c_smiles_tokens  : {smiles_tokens.shape}, dtype={smiles_tokens.dtype}")
    print(f"c_fingerprints   : {fingerprints.shape},  dtype={fingerprints.dtype}")
    print(f"c_graph_feat     : {graph_feat.shape},    dtype={graph_feat.dtype}")
    print(f"c_graph_adj      : {graph_adj.shape},     dtype={graph_adj.dtype}")


if __name__ == "__main__":
    main()
"""
generate_esm2_embeddings.py
===========================================================
用 ESM2 对蛋白质序列进行编码，生成 [N_p, 1280] 的嵌入矩阵。

输入文件:
  - plist_sequences.csv        蛋白质序列表
  - plist_full.txt             蛋白质 ID 列表（决定 npy 输出的行序）

输出文件:
  - protein_esm2_emb.npy       形状 [7258, 1280]，行序与 plist_full.txt 一致

ESM2 模型选择（按显存从小到大）:
  esm2_t6_8M_UR50D       →  320 维  ~0.1GB
  esm2_t12_35M_UR50D     →  480 维  ~0.2GB
  esm2_t30_150M_UR50D    →  640 维  ~0.6GB
  esm2_t33_650M_UR50D    → 1280 维  ~2.5GB  ← 默认（与 HyperCTI 对接）
  esm2_t36_3B_UR50D      → 2560 维  ~12GB
  esm2_t48_15B_UR50D     → 5120 维  ~60GB

长序列处理（seq_len > 1022）:
  采用滑动窗口均值策略，窗口=1022，步长=512
  比直接截断保留更完整的序列信息

依赖安装:
  pip install fair-esm torch numpy pandas
===========================================================
"""

import os
import time
import numpy as np
import pandas as pd
import torch

# ──────────────────────────────────────────────────────────────────
# 配置区（按需修改）
# ──────────────────────────────────────────────────────────────────
# 自动定位项目根目录
dataset_type = "Symmap"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, f"datasets/{dataset_type}/targets.csv")
PLIST_PATH = os.path.join(BASE_DIR, f"datasets/{dataset_type}/target_ids.txt")
OUTPUT_PATH = os.path.join(BASE_DIR, f"datasets/{dataset_type}/features/newfeatures/protein_esm2_emb.npy")

MODEL_NAME    = 'esm2_t33_650M_UR50D'  # 输出 1280 维
REPR_LAYER    = 33                      # 取最后一层
EMB_DIM       = 1280

BATCH_SIZE    = 4      # 短序列批大小（长序列显存不足时调小）
WINDOW_SIZE   = 1022   # ESM2 最大输入氨基酸数
WINDOW_STRIDE = 512    # 滑动窗口步长
DEVICE        = 'cuda' if torch.cuda.is_available() else 'cpu'
# ──────────────────────────────────────────────────────────────────


def encode_long_seq(model, batch_converter, pid, seq, device, repr_layer,
                    window_size, stride):
    """滑动窗口分段编码超长序列，取各窗口 mean pooling 后再均值。"""
    seqlen = len(seq)
    windows, start = [], 0
    while start < seqlen:
        end = min(start + window_size, seqlen)
        windows.append(seq[start:end])
        if end == seqlen:
            break
        start += stride

    embs = []
    for w in windows:
        _, _, tokens = batch_converter([(pid, w)])
        tokens = tokens.to(device)
        with torch.no_grad():
            out = model(tokens, repr_layers=[repr_layer], return_contacts=False)
        # 去掉首尾特殊 token，对有效位置取均值
        emb = out['representations'][repr_layer][0, 1:len(w)+1].mean(0)
        embs.append(emb.cpu().float().numpy())
    return np.mean(embs, axis=0)  # [EMB_DIM]


def encode_short_batch(model, batch_converter, ids, seqs, device, repr_layer):
    """批量编码短序列（≤ 1022），返回 {id: emb} 字典。"""
    data = list(zip(ids, seqs))
    _, _, tokens = batch_converter(data)
    tokens = tokens.to(device)
    with torch.no_grad():
        out = model(tokens, repr_layers=[repr_layer], return_contacts=False)
    token_repr = out['representations'][repr_layer]
    result = {}
    for j, (bid, bseq) in enumerate(data):
        emb = token_repr[j, 1:len(bseq)+1].mean(0)
        result[bid] = emb.cpu().float().numpy()
    return result


def main():
    print("=" * 60)
    print("ESM2 蛋白质序列编码")
    print("=" * 60)

    # ── 1. 加载序列 CSV ──────────────────────────────────────────
    print(f"\n[1/5] 加载序列文件: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    df.dropna(subset=["sequence"], inplace=True)
    df['seq_len'] = df['sequence'].str.len()

    if dataset_type == "Symmap":
        # Symmap
        df.rename(columns={"protein_id": "target_id"}, inplace=True)
    elif dataset_type == "TCMsuite":
        # TCMsuite
        df.rename(columns={"id1": "target_id"}, inplace=True)
    else:
        raise ValueError("please select a correct dataset type.")

    df.dropna(subset="sequence", inplace=True) 
    df['target_id'] = df['target_id'].astype(str)
    id2seq  = dict(zip(df['target_id'], df['sequence']))


    total = len(df)
    has_seq = df['sequence'].notna().sum()
    print(f"  总蛋白质数: {total}")

    # 序列长度分布
    long_seqs = df[df['seq_len'] > WINDOW_SIZE]
    print(f"\n  序列长度 > {WINDOW_SIZE}（需滑动窗口）: {len(long_seqs)} 条")
    print(f"  序列长度 ≤ {WINDOW_SIZE}（直接编码）:  {has_seq - len(long_seqs)} 条")

    # ── 2. 加载 plist_full（决定输出行序） ───────────────────────
    print(f"\n[2/5] 加载 plist_full: {PLIST_PATH}")
    with open(PLIST_PATH, 'r') as f:
        plist = [ln.strip() for ln in f if ln.strip()]
    print(f"  plist_full 蛋白质数: {len(plist)}")

    missing = set(plist) - set(df['target_id'])
    if missing:
        print(f"  ⚠ 有 {len(missing)} 个 target_id 在 CSV 中不存在，将用零向量填充")

    plist_index = {pid: i for i, pid in enumerate(plist)}

    # ── 3. 加载 ESM2 ─────────────────────────────────────────────
    print(f"\n[3/5] 加载模型: {MODEL_NAME}")
    print(f"  设备: {DEVICE}")
    import esm
    model, alphabet = getattr(esm.pretrained, MODEL_NAME)()
    batch_converter = alphabet.get_batch_converter()
    model = model.to(DEVICE)
    model.eval()
    print(f"  加载完成，输出维度: {EMB_DIM}")

    # ── 4. 编码 ──────────────────────────────────────────────────
    print(f"\n[4/5] 开始编码 {len(plist)} 条序列...")
    print(f"  批大小={BATCH_SIZE}  窗口={WINDOW_SIZE}  步长={WINDOW_STRIDE}")

    emb_matrix = np.zeros((len(plist), EMB_DIM), dtype=np.float32)
    t0 = time.time()

    pending_ids  = []
    pending_seqs = []
    done = long_count = zero_count = 0

    def flush_batch():
        if not pending_ids:
            return
        result = encode_short_batch(
            model, batch_converter,
            pending_ids, pending_seqs,
            DEVICE, REPR_LAYER)
        for bid, emb in result.items():
            emb_matrix[plist_index[bid]] = emb
        pending_ids.clear()
        pending_seqs.clear()

    for pid in plist:
        raw = id2seq.get(pid, None)

        # 无序列：零向量
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            emb_matrix[plist_index[pid]] = np.zeros(EMB_DIM, dtype=np.float32)
            zero_count += 1
            done += 1
            continue

        seq = str(raw).strip()

        if len(seq) > WINDOW_SIZE:
            # 先清空短序列批
            flush_batch()
            # 滑动窗口
            emb = encode_long_seq(
                model, batch_converter, pid, seq,
                DEVICE, REPR_LAYER, WINDOW_SIZE, WINDOW_STRIDE)
            emb_matrix[plist_index[pid]] = emb
            long_count += 1
        else:
            pending_ids.append(pid)
            pending_seqs.append(seq)
            if len(pending_ids) >= BATCH_SIZE:
                flush_batch()

        done += 1
        if done % 200 == 0 or done == len(plist):
            elapsed = time.time() - t0
            speed   = done / elapsed
            eta     = (len(plist) - done) / speed if speed > 0 else 0
            print(f"  [{done:5d}/{len(plist)}]  "
                  f"耗时 {elapsed:6.0f}s  "
                  f"速度 {speed:.1f}条/s  "
                  f"剩余 ~{eta:.0f}s")

    flush_batch()  # 处理最后一批

    elapsed = time.time() - t0
    print(f"\n  ✅ 编码完成")
    print(f"  总耗时:         {elapsed:.1f}s  ({elapsed/60:.1f}min)")
    print(f"  长序列（窗口）: {long_count} 条")
    print(f"  零向量填充:     {zero_count} 条")

    # ── 5. 保存并验证 ────────────────────────────────────────────
    print(f"\n[5/5] 保存: {OUTPUT_PATH}")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    np.save(OUTPUT_PATH, emb_matrix)

    # 验证
    loaded = np.load(OUTPUT_PATH)
    nonzero = (np.abs(loaded).sum(axis=1) > 0).sum()
    size_mb = os.path.getsize(OUTPUT_PATH) / 1024 / 1024

    print(f"\n{'='*60}")
    print(f"✅ 全部完成！")
    print(f"  输出路径:   {OUTPUT_PATH}")
    print(f"  矩阵形状:   {loaded.shape}  dtype={loaded.dtype}")
    print(f"  文件大小:   {size_mb:.1f} MB")
    print(f"  非零行数:   {nonzero}/{len(plist)}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
    
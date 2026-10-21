import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              f1_score, recall_score, precision_score,
                              precision_recall_curve, roc_curve)
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

files = [
      "2026-05-10 17-45-30[1].txt",
    "2026-05-10 19-48-06[2].txt",
    "2026-05-10 21-58-28[3].txt",
    "2026-05-11 00-03-59[4].txt",
    "2026-05-11 02-03-06[5].txt"
]

# 脚本在 tools/ 下，往上一级即 HyperCTI-main4-1/，再进入 results/cv/
cv_path = os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'results', 'cv'))

all_auc, all_aupr, all_f1, all_recall, all_precision = [], [], [], [], []

print(f"{'Fold':<10} {'AUC':<10} {'AUPR':<10} {'Recall':<10} {'F1':<10} {'Precision':<10} {'Best_Thr':<10}")
print("-" * 70)

for i, f in enumerate(files):
    full_path = os.path.join(cv_path, f)
    data = pd.read_csv(full_path, header=None)
    y_true   = data[0].values.astype(int)
    y_scores = data[1].values.astype(float)

    auc  = roc_auc_score(y_true, y_scores)
    aupr = average_precision_score(y_true, y_scores)

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    f1_scores = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-8)
    best_idx  = np.argmax(f1_scores)
    best_thr  = thresholds[best_idx]

    y_pred    = (y_scores >= best_thr).astype(int)
    f1        = f1_score(y_true, y_pred)
    recall    = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)

    all_auc.append(auc);  all_aupr.append(aupr)
    all_f1.append(f1);    all_recall.append(recall);  all_precision.append(precision)

    print(f"Fold {i+1:<5} {auc:<10.4f} {aupr:<10.4f} {recall:<10.4f} {f1:<10.4f} {precision:<10.4f} {best_thr:<10.4f}")

print("-" * 70)
print(f"{'Average':<10} {np.mean(all_auc):<10.4f} {np.mean(all_aupr):<10.4f} "
      f"{np.mean(all_recall):<10.4f} {np.mean(all_f1):<10.4f} {np.mean(all_precision):<10.4f}")
print(f"{'Std Dev':<10} {np.std(all_auc):<10.4f} {np.std(all_aupr):<10.4f} "
      f"{np.std(all_recall):<10.4f} {np.std(all_f1):<10.4f} {np.std(all_precision):<10.4f}")
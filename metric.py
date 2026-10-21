
import os
import pandas as pd
import numpy as np
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              f1_score, recall_score, precision_score,
                              precision_recall_curve)

def metric(y_true, y_pred):

    auc = roc_auc_score(y_true, y_pred)
    aupr = average_precision_score(y_true, y_pred)

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred)
    beta = 1
    f_beta = ((1 + beta**2) * precisions[:-1] * recalls[:-1] /
              (beta**2 * precisions[:-1] + recalls[:-1] + 1e-8))
    best_idx  = np.argmax(f_beta)
    best_thr  = thresholds[best_idx]

    y_pred    = (y_pred >= best_thr).astype(int)
    f1        = f1_score(y_true, y_pred)
    recall    = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)

    return auc, aupr, precision, recall, f1


if __name__ == "__main__":
    
    d_types = ["Symmap", "TCMsuite"]
    fold_result_dir = "experiment/0902_fusion/results/ablation/without_f"

    for d in d_types:
        dataset_dir = os.path.join(fold_result_dir, d, "cv")
        if not os.path.exists(dataset_dir):
            continue
        print(f"\n======== Dataset Type: {d} ========")
        
        # 每个数据集单独初始化列表，用于收集 5 个 fold 的结果
        auc_list, aupr_list, precision_list, recall_list, f1_list = [], [], [], [], []
        
        # 遍历该数据集下的所有 fold 预测结果文件
        for file_name in os.listdir(dataset_dir):
        # for fold in range(5):
            # file_path = os.path.join(dataset_dir, f"fold_{str(fold)}", "result.csv")
            
            # HyperAttentionCTI
            # file_path = os.path.join(dataset_dir, str(fold), f"{d}_Test_stable_prediction copy.txt")
            
            # 过滤只读取 csv 文件，防止读取到隐藏文件（如 .DS_Store）
            # PerceiverCTI
            # if not (file_name.startswith("fold")  and file_name.endswith('.csv')):
            #     continue
            
            # # HGNNLDA
            # if not file_name.endswith(".txt"):
            #     continue
                
            # data = pd.read_csv(file_path) 
            data = pd.read_csv(os.path.join(dataset_dir, file_name)) 
            # data = pd.read_csv(file_path, sep=" ", header=None) 
            
            # 计算当前 fold 的指标
            auc, aupr, precision, recall, f1 = metric(
                data["label"].values.tolist(),
                data["predict"].values.tolist()
            )
            
            # 保存到当前数据集的列表中
            auc_list.append(auc)
            aupr_list.append(aupr)
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)
        
        # 计算并打印当前数据集 5-fold 的 Mean ± Std
        metrics_map = {
            "AUC": auc_list,
            "AUPR": aupr_list,
            "F1": f1_list,
            "Precision": precision_list,
            "Recall": recall_list
        }
        
        for metric_name, values in metrics_map.items():
            if len(values) > 0:
                mean_val = np.mean(values)
                std_val = np.std(values) # 如果需要无偏样本标准差，可改用 np.std(values, ddof=1)
                print(f"{metric_name:<10}: {mean_val:.5f} ± {std_val:.5f}")
            else:
                print(f"{metric_name:<10}: No valid fold files found.")
                
        print("=" * 40)
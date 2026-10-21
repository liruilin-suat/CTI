import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def disease_aware_20_diseases_5fold(
        df_cpi: pd.DataFrame,
        df_td: pd.DataFrame,
        seed: int=42,
        disjoint: bool=True,
        save_dir: str = None
):
    """5-Fold 划分协议：每一折固定抽取 20 个 Disease 作为测试集

    参数:
        disjoint (bool):
            True: 从 Disease 中抽取 100 个，均匀分成 5 折（每折 20 个，互不重叠）
            False: 独立随机抽 5 次，每次随机抽 20 个 Disease
    """
    np.random.seed(seed)

    all_diseases = df_td['disease_id'].unique()
    all_targets = set(df_cpi['target_id'].unique())

    if len(all_diseases) < 100 and disjoint:
        raise ValueError('Disease 总数小于 100，无法分成 5 折互不重叠的 20 个 Disease！')

    # 生成每一折的测试 Disease 列表
    fold_test_diseases = []
    if disjoint:
        # 从全量 Disease 中不放回抽取 100 个 Disease，并分成 5 组
        selected_100 = np.random.choice(all_diseases, size=100, replace=False)
        fold_test_diseases = [
            set(selected_100[i * 20 : (i + 1) * 20]) for i in range(5)
        ]
    else:
        # 5 次独立抽取，每次 20 个
        for _ in range(5):
            fold_test_diseases.append(
                set(np.random.choice(all_diseases, size=20, replace=False))
            )

    folds_data = []

    print(
        f'=== 开始划分（每一折测试集固定 20 个 Disease，Seed: {seed}）===\n'
    )

    for fold, test_diseases in enumerate(fold_test_diseases):
        train_diseases = set(all_diseases) - test_diseases

        # 提取当前 Fold 20 个测试 Disease 关联的所有 Target
        test_targets = set(
            df_td[df_td['disease_id'].isin(test_diseases)]['target_id'].unique()
        )

        # 严格隔离：训练集 Target 彻底剔除 test_targets
        train_targets = all_targets - test_targets

        # 3. 划分 CPI 数据 [compound_id, target_id, label]
        test_cpi = df_cpi[df_cpi['target_id'].isin(test_targets)].copy()
        train_cpi = df_cpi[df_cpi['target_id'].isin(train_targets)].copy()

        # 划分训练集超图边数据 [target_id, disease_id]
        train_td_edges = df_td[
            (df_td['disease_id'].isin(train_diseases))
            & (df_td['target_id'].isin(train_targets))
        ].copy()

        test_td_edges = df_td[
            ~((df_td['disease_id'].isin(train_diseases))
            & (df_td['target_id'].isin(train_targets)))
        ].copy()

        fold_info = {
            'fold': fold + 1,
            'test_diseases': list(test_diseases),
            'test_targets': list(test_targets),
        }
        folds_data.append(fold_info)

        # save
        save = save_dir + f"/exp2/"
        os.makedirs(save, exist_ok=True)
        train_cpi.to_csv(save + f"/train_fold_{fold}.txt", sep="\t", index=False, header=False)
        test_cpi.to_csv(save + f"/test_fold_{fold}.txt", sep="\t", index=False, header=False)
        train_td_edges.to_csv(save + f"/target_disease_train_{fold}.txt", sep="\t",
                              index=False, header=False)
        
        test_td_edges.to_csv(save + f"/target_disease_test_{fold}.txt", sep="\t",
                              index=False, header=False)

        print(f'Fold {fold+1}:')
        print(f'  - 测试集 Disease 数量: {len(test_diseases)}')
        print(f'  - 测试集关联 Target 数量: {len(test_targets)}')
        print(
            f'  - 训练集 CPI 样本数: {len(train_cpi)} | 测试集 CPI 样本数: {len(test_cpi)}'
        )
        print(
            f'  - 训练/测试比例: {round(len(train_cpi) / (len(train_cpi) + len(test_cpi)) * 100, 2)}% / '
            f'{round(len(test_cpi) / (len(train_cpi) + len(test_cpi)) * 100, 2)}%\n'
        )

    return folds_data

def np_encoder(obj):
    if isinstance(obj, np.generic):
        return obj.item()  # 将 np.int64/float64 等转为 Python 原生 int/float
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # 将 NumPy 数组转为 Python list
    if isinstance(obj, set):
        return list(obj)  # 顺便兼容 set
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

if __name__ == '__main__':

    # 假设数据格式：
    # df_cpi: [compound_id, target_id, label]
    # df_td:  [target_id, disease_id]
    # df_hc:  [herb_id, compound_id]

    save_dir = "./datasets/{}"
    for i, dataset_type in enumerate(["Symmap", "TCMsuite"]):
        # ================= 数据读取与预处理 =================
        cti_file = f'./datasets/{dataset_type}/train_datasets/one_zero.txt'
        td_file = f'./datasets/{dataset_type}/target_disease_cleaned.txt'

        df_cpi = pd.read_csv(cti_file, sep='\t', header=None)
        df_td = pd.read_csv(td_file, sep='\t', header=None)

        df_cpi.rename(
            columns={0: 'compound_id', 1: 'target_id', 2: 'label'}, inplace=True
        )
        df_td.rename(columns={0: 'target_id', 1: 'disease_id'}, inplace=True)

        df_target_unique = pd.DataFrame({'target_id': df_td['target_id'].unique()})
        df_td = pd.merge(
            df_td, df_target_unique[['target_id']], how='inner', on=['target_id']
        )
        folds = disease_aware_20_diseases_5fold(df_cpi, df_td, seed=42,
                                                save_dir=save_dir.format(dataset_type))
        # 打开文件并写入 JSON 数据
        with open(save_dir.format(dataset_type) + '/exp2/disease_split.json', 'w', encoding='utf-8') as f:
            json.dump(folds, f, default=np_encoder, ensure_ascii=False, indent=4)
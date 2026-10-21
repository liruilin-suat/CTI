import itertools
import os

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.cluster import KMeans


def cluster_diseases_by_tfidf(
    disease_texts: list[str],
    n_clusters: int = 5,
    max_features: int = 64,
    random_state: int = 42,
) -> pd.DataFrame:
    """使用 TF-IDF 和 K-Means 对疾病文本数据进行特征提取与分类/分折 (Folds)。

    参数:
        disease_texts (list[str'): 疾病名称列的名称
        n_clusters (int): K-Means 聚类簇数 (默认 5)
        max_features (int): TF-IDF 提取的最大特征数 (默认 64)
        random_state (int): 随机种子，确保结果可复现

    返回:
        pd.DataFrame: 包含聚类类别 ('class') 的 DataFrame。
    """
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        lowercase=True,
        max_features=max_features,
    )
    disease_tfidf_matrix = vectorizer.fit_transform(disease_texts)

    # K-Means 聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    disease_clusters = kmeans.fit_predict(disease_tfidf_matrix)

    return disease_clusters

if __name__ == '__main__':

    # disease_path = "/home/liruilin/SUAT/PyProject/hypergraph/synergy/data/TCM/symmap/raw/disease.xlsx"
    #
    # disease_df = pd.read_excel(disease_path)
    #
    # # TCMsuite
    # # disease_df = disease_df[["id1", "name", "xbDiseaseName"]]
    # # Symmap
    # disease_df = disease_df[["Disease_id", "Disease_Name"]]
    #
    # disease_ids_path = "/home/liruilin/SUAT/PyProject/hypergraph/HyperCTI/HyperCTI-main/datasets/Symmap/disease_id_all.txt"
    # disease_ids = pd.read_csv(disease_ids_path, header=None)
    # # TCMsuite
    # # disease_ids.rename(columns={0: "id1"}, inplace=True)
    # # Symmap
    # disease_ids.rename(columns={0: "Disease_id"}, inplace=True)
    # print(disease_df.shape)
    # disease_df = pd.merge(disease_df, disease_ids, on="Disease_id")
    # # TCMsuite
    # # disease_df["disease_name"] = disease_df["name"].combine_first(disease_df["xbDiseaseName"])
    #
    # # TCMsuite
    # # disease_texts = disease_df["disease_name"].values.tolist()
    # # Symmap
    # disease_texts = disease_df["Disease_Name"].values.tolist()
    #
    # disease_clusters = cluster_diseases_by_tfidf(disease_texts)
    #
    # disease_df["class"] = disease_clusters
    #
    # disease_df.to_csv("Symmap_disease_class.csv", index=False)
    #
    # # 统计每个 Cluster 的数量
    # counts = pd.Series(disease_clusters).value_counts().sort_index()
    #
    # # 格式化输出
    # print("--- 各疾病类别（Fold）样本数统计 ---")
    # for cluster_id, count in counts.items():
    #     percentage = (count / len(disease_clusters)) * 100
    #     print(f"Cluster {cluster_id}: {count} 个疾病 ({percentage:.2f}%)")

    # exp3 dataset split -- 5-cv
    dataset = "TCMsuite"
    datasets_dir = "/home/liruilin/SUAT/PyProject/hypergraph/HyperCTI/HyperCTI-main/datasets"

    save_dir = "./exp3"
    os.makedirs(save_dir, exist_ok=True)

    cti_file = f'{datasets_dir}/{dataset}/train_datasets/one_zero.txt'
    td_file = f'{datasets_dir}/{dataset}/target_disease_cleaned.txt'
    hc_file = f'{datasets_dir}/{dataset}/herb_compound_cleaned.txt'
    hd_file = f'{datasets_dir}/{dataset}/herb_disease.txt'

    cti_df = pd.read_csv(cti_file, sep='\t', header=None)
    td_df = pd.read_csv(td_file, sep='\t', header=None)
    hc_df = pd.read_csv(hc_file, sep='\t', header=None)
    hd_df = pd.read_csv(hd_file, sep='\t', header=None)
    all_cti_set = set(zip(cti_df[0], cti_df[1]))

    all_disease_ids = set(td_df[1].unique())
    disease_class_file = "./{}_disease_class.csv"
    c_df = pd.read_csv(disease_class_file.format(dataset))
    c_df.rename(columns={"id1": 1}, inplace=True)
    for cls in range(5):
        one_cls = c_df[c_df['class'] == cls]
        disease_ids = pd.merge(one_cls, td_df, on=1)
        disease_ids_df = pd.DataFrame({1 : disease_ids[1].unique()})

        test_disease_ids = set(disease_ids[1].unique())

        herb_ids = pd.merge(disease_ids_df, hd_df, on=1)
        herb_ids_df = pd.DataFrame({0 : herb_ids[0].unique()})

        compound_ids = pd.merge(herb_ids[[0]], hc_df, on=0)[1].unique().tolist()
        target_ids = disease_ids[0].unique().tolist()

        # train and test
        test_mask = cti_df[0].isin(compound_ids) & cti_df[1].isin(target_ids)

        # 3. 划分测试集与训练集 (完美保留完整的 DataFrame 列结构)
        test_cti_df = cti_df[test_mask].copy()
        train_cti_df = cti_df[~test_mask].copy()

        train_disease_ids = all_disease_ids - test_disease_ids
        test_td_edge = td_df[td_df[1].isin(test_disease_ids)]
        train_td_edge = td_df[td_df[1].isin(train_disease_ids)]

        print(f"全量 CTI 数量: {len(all_cti_set)}")
        print(f"测试集 CTI 数量: {len(test_cti_df)}")
        print(f"训练集 CTI 数量: {len(train_cti_df)}")
        print(f"占比: {len(test_cti_df) / len(all_cti_set) * 100 :.4}")

        train_cti_df.to_csv(os.path.join(save_dir, f"train_fold_{cls}.txt"),
                            sep="\t", header=False, index=False)
        test_cti_df.to_csv(os.path.join(save_dir, f"test_fold_{cls}.txt"),
                           sep="\t", header=False, index=False)
        train_td_edge.to_csv(os.path.join(save_dir, f"target_disease_cleaned_train_{cls}.txt"),
                             sep="\t", header=False, index=False)
        test_td_edge.to_csv(os.path.join(save_dir, f"target_disease_cleaned_test_{cls}.txt"),
                             sep="\t", header=False, index=False)
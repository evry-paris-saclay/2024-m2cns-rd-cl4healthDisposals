import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist, squareform
import seaborn as sns


def log_expected_empirical_prediction(predictions: np.ndarray, labels: np.ndarray):
    """
    Log Expected Empirical Prediction (LEEP) as described in ICML 2020.

    Args:
        predictions (np.ndarray): Predictions of the pre-trained model (N, C_s).
        labels (np.ndarray): Ground-truth labels (N,).

    Returns:
        float: LEEP score.
    """
    N, C_s = predictions.shape
    labels = labels.reshape(-1)
    C_t = int(np.max(labels) + 1)

    # Ensure predictions are normalized probabilities
    predictions = predictions / predictions.sum(axis=1, keepdims=True)
    #print(predictions.shape)
    #print(labels.shape)

    # Joint distribution (P(y, z))
    joint = np.zeros((C_t, C_s), dtype=float)
    for i in range(C_t):
        this_class = predictions[labels == i]
        joint[i] = np.sum(this_class, axis=0)

    # Conditional probability (P(y | z))
    with np.errstate(divide='ignore', invalid='ignore'):
        p_target_given_source = np.nan_to_num(joint / joint.sum(axis=0, keepdims=True)).T

    # Compute empirical prediction
    empirical_prediction = predictions @ p_target_given_source
    empirical_prob = np.array([predict[label] for predict, label in zip(empirical_prediction, labels)])

    # Clip to avoid invalid log values
    epsilon = 1e-10  # A small constant for numerical stability
    empirical_prob = np.clip(empirical_prob, epsilon, 1.0)
    # print(empirical_prob)

    # Compute LEEP score
    score = np.mean(np.log(empirical_prob))
    return score


def plot_distance_matrix_leep(leep_scores, dataset_names):
    # 计算任务间距离
    distance_matrix = pdist(np.array(leep_scores).reshape(-1, 1), metric='euclidean')
    distance_matrix_full = squareform(distance_matrix)  # 转换为对称矩阵形式
    linkage_matrix = linkage(distance_matrix, method='complete', optimal_ordering=True)
    # 使用 clustermap 绘制距离矩阵和层次聚类
    sns.clustermap(
        distance_matrix_full,
        row_linkage=linkage_matrix,  # 行聚类
        col_linkage=linkage_matrix,  # 列聚类
        cmap='viridis',             # 颜色映射
        xticklabels=dataset_names,  # 横轴标签
        yticklabels=dataset_names,  # 纵轴标签
        figsize=(10, 8)             # 图像尺寸
    )
    plt.title("Hierarchical Clustering and Distance Matrix")
    plt.show()
    # 绘制距离矩阵
    # plt.figure(figsize=(8, 6))
    # plt.imshow(distance_matrix, cmap='viridis', interpolation='nearest')
    # plt.colorbar(label="Distance")
    # plt.xticks(range(len(dataset_names)), dataset_names, rotation=90)
    # plt.yticks(range(len(dataset_names)), dataset_names)
    # plt.title("Distance Matrix based on LEEP Scores")
    # plt.show()
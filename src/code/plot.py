import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import squareform
from sklearn.metrics import confusion_matrix

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage

# matplotlib.use('TkAgg')
# matplotlib.use('MacOSX')  # Using this if in MacOS


def plot_metrics(epochs, train_values, val_values, ylabel, title, train_label="Training", val_label="Validation", save_path=None, ylim=None, ymin=None, ymax=None):
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_values, label=f"{train_label} {ylabel}", marker='o')
    plt.plot(epochs, val_values, label=f"{val_label} {ylabel}", marker='o')
    plt.xlabel("Epochs")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid()

    if ylim:
        plt.ylim(ylim)

    if ymin:
        plt.ylim(bottom=ymin)

    plt.show()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")


def plot_metric_continue_train(epochs, train_values_t1, train_values_t2, train_values_t3, ylabel, title, save_path=None, ylim=None, ymin=None, ymax=None):
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_values_t1, label=f"Task1 {ylabel}", marker='o')
    plt.plot(epochs, train_values_t2, label=f"Task2 {ylabel}", marker='o')
    plt.plot(epochs, train_values_t3, label=f"Task3 {ylabel}", marker='o')
    plt.xlabel("Epochs")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid()

    if ylim:
        plt.ylim(ylim)

    if ymin:
        plt.ylim(bottom=ymin)

    plt.show()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")


def plot_metric_continue_evalu(task, evalu_values_t1, evalu_values_t2, evalu_values_t3, ylabel, title, save_path=None, ylim=None, ymin=None, ymax=None):
    plt.figure(figsize=(10, 5))
    evalu_values_t2_modified = evalu_values_t2.copy()
    evalu_values_t2_modified[0] = np.nan
    evalu_values_t3_modified = evalu_values_t3.copy()
    evalu_values_t3_modified[0] = np.nan
    evalu_values_t3_modified[1] = np.nan
    plt.plot(task, evalu_values_t1, label=f"Task 1 {ylabel}", marker='o')
    plt.plot(task, evalu_values_t2_modified, label=f"Task 2 {ylabel}", marker='o')
    plt.plot(task, evalu_values_t3_modified, label=f"Task 3 {ylabel}", marker='o')
    plt.xlabel("Version of Model")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.xticks(np.arange(1, 4, 1))

    if ylim:
        plt.ylim(ylim)

    if ymin:
        plt.ylim(bottom=ymin)

    plt.show()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")


def plot_confusion_matrix(all_labels, all_predictions, class_names):
    cm = confusion_matrix(all_labels, all_predictions, labels=np.arange(len(class_names)))
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.show()


# matrix euclid
def plot_similarity_matrix(class_centroids,  title="Class Similarity Matrix", save_path="/home/hungry_gould/projet/2024-m2cns-rd-cl4healthDisposals/src/results/euclid_similarity_matrix.png"):
    class_names = list(class_centroids.keys())
    centroids = np.array(list(class_centroids.values()))

    distance_matrix = euclidean_distances(centroids, centroids)
    condensed_distance = squareform(distance_matrix)
    max_distance = np.max(distance_matrix)
    similarity_matrix = 1 - (distance_matrix / max_distance)
    linkage_matrix = linkage(condensed_distance, method='complete', optimal_ordering=True)

    similarity_df = pd.DataFrame(similarity_matrix, index=class_names, columns=class_names)
    cluster_grid = sns.clustermap(similarity_df,
                                  # row_cluster=False,
                                  # col_cluster=True,
                                  row_linkage=linkage_matrix,
                                  col_linkage=linkage_matrix,
                                  cmap='viridis_r')
    # cluster_grid.ax_heatmap.set_title(title)
    cluster_grid.ax_heatmap.set_xlabel("Classes")
    cluster_grid.ax_heatmap.set_ylabel("Classes")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_distance_matrix(class_centroids, title="Class Distance Matrix", save_path="/home/hungry_gould/projet/2024-m2cns-rd-cl4healthDisposals/src/results/euclid_distance_matrix.png"):
    class_names = list(class_centroids.keys())
    centroids = np.array(list(class_centroids.values()))

    distance_matrix = euclidean_distances(centroids, centroids)
    condensed_distance = squareform(distance_matrix)
    linkage_matrix = linkage(condensed_distance, method='complete', optimal_ordering=True)


    distance_df = pd.DataFrame(distance_matrix, index=class_names, columns=class_names)
    cluster_grid = sns.clustermap(distance_df,
                                  # row_cluster=False,
                                  # col_cluster=True,
                                  row_linkage=linkage_matrix,
                                  col_linkage=linkage_matrix,
                                  cmap='viridis_r')
    # cluster_grid.ax_heatmap.set_title(title)
    cluster_grid.ax_heatmap.set_xlabel("Classes")
    cluster_grid.ax_heatmap.set_ylabel("Classes")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


# silhouette euclid
def plot_determine_optimal_clusters_silhouette(class_centroids, max_clusters=12, save_path="/home/hungry_gould/projet/2024-m2cns-rd-cl4healthDisposals/src/results/euclid_silhouette.png"):
    centroids = np.array(list(class_centroids.values()))
    distance_matrix = euclidean_distances(centroids, centroids)
    silhouette_scores = []
    for n_clusters in range(2, max_clusters + 1):
        clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage='average')
        labels = clustering.fit_predict(distance_matrix)
        score = silhouette_score(distance_matrix, labels, metric='precomputed')
        silhouette_scores.append(score)

    # Plot silhouette scores
    plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Method')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

    # Return the optimal number of clusters
    return np.argmax(silhouette_scores) + 2  # +2 because range starts from 2


def determine_optimal_clusters(class_centroids, max_clusters=10):
    centroids = np.array(list(class_centroids.values()))
    sse = []
    for n_clusters in range(2, max_clusters + 1):
        # Perform KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(centroids)
        # Compute SSE using KMeans inertia
        sse.append(kmeans.inertia_)

    # Plot the elbow curve
    plt.plot(range(2, max_clusters + 1), sse, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('SSE')
    plt.title('Elbow Method')
    plt.show()


# silhouette task2vec
def plot_silhouette_scores(silhouette_scores, max_clusters=5, save_path="/home/hungry_gould/projet/2024-m2cns-rd-cl4healthDisposals/src/results/task2vec_silhouette.png"):
    plt.figure(figsize=(8, 6))
    plt.plot(list(silhouette_scores.keys()), list(silhouette_scores.values()), marker='o')
    # plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
    plt.title("Silhouette Score")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_leep_scores_ligne(leep_scores, save_path="/home/hungry_gould/projet/2024-m2cns-rd-cl4healthDisposals/src/results/plot_leep_scores_ligne.png"):
    from sklearn.linear_model import LinearRegression

    sorted_leep_scores = sorted(leep_scores.items(), key=lambda x: x[1], reverse=True)

    labels = [f"{source} -> {target}" for (source, target), score in sorted_leep_scores]
    values = [score for _, score in sorted_leep_scores]

    x = np.arange(len(values)).reshape(-1, 1)  # X 坐标 (任务索引)
    y = np.array(values)  # Y 坐标 (LEEP 分数)

    model = LinearRegression()
    model.fit(x, y)
    y_pred = model.predict(x)  # 拟合直线的预测值

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker='o', linestyle='-', label="LEEP Scores")
    plt.plot(x, y_pred, color='red', linestyle='--', label="Fitted Line (Linear Regression)")

    for i, (label, value) in enumerate(zip(labels, values)):
        plt.text(i, value, f"{value:.2e}", fontsize=8, ha='center', va='bottom')

    plt.title("Sorted LEEP Scores with Fitted Line")
    plt.xlabel("Task Transitions")
    plt.ylabel("LEEP Score")
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.grid(alpha=0.5)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

    

def plot_leep_scores_heatmap(leep_scores,task_names,save_path="/home/hungry_gould/projet/2024-m2cns-rd-cl4healthDisposals/src/results/plot_leep_score_hotmap.png"):
    
    plt.figure(figsize=(10, 6))
    # 构造矩阵
    score_matrix = pd.DataFrame(index=task_names, columns=task_names)
    for (source, target), score in leep_scores.items():
        score_matrix.loc[source, target] = score

    sns.heatmap(score_matrix.astype(float), annot=True, fmt=".4f", cmap="coolwarm")
    plt.title("LEEP Scores Between Tasks")
    plt.xlabel("Target Task")
    plt.ylabel("Source Task")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_leep_scores_order(leep_scores, output ,save_path="/home/hungry_gould/projet/2024-m2cns-rd-cl4healthDisposals/src/results/plot_leep_scores_order.png"):
    from sklearn.linear_model import LinearRegression

    sorted_leep_scores = sorted(leep_scores.items(), key=lambda x: x[1], reverse=True)

    labels = [f"{source} -> {target}" for (source, target), score in sorted_leep_scores]
    values = [score for _, score in sorted_leep_scores]

    x = np.arange(len(values)).reshape(-1, 1)  # X 坐标 (任务索引)
    y = np.array(values)  # Y 坐标 (LEEP 分数)

    # model = LinearRegression()
    # model.fit(x, y)
    # y_pred = model.predict(x)  # 拟合直线的预测值

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker='o', linestyle='-', label="LEEP Scores")
    # plt.plot(x, y_pred, color='red', linestyle='--', label="Fitted Line (Linear Regression)")
    plt.plot(x, y, label=output)

    for i, (label, value) in enumerate(zip(labels, values)):
        plt.text(i, value, f"{value:.2e}", fontsize=8, ha='center', va='bottom')

    plt.title("Sorted LEEP Scores with order")
    plt.xlabel("Task Transitions")
    plt.ylabel("LEEP Score")
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.grid(alpha=0.5)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()



def plot_metric_continue_train_V2(epochs, train_values_t1, train_values_t2, train_values_t3, train_values_t4, ylabel, title, save_path=None, ylim=None, ymin=None, ymax=None):
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_values_t1, label=f"Task4 {ylabel}", marker='o')
    plt.plot(epochs, train_values_t2, label=f"Task2 {ylabel}", marker='o')
    plt.plot(epochs, train_values_t3, label=f"Task3 {ylabel}", marker='o')
    plt.plot(epochs, train_values_t4, label=f"Task1 {ylabel}", marker='o')
    plt.xlabel("Epochs")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid()

    if ylim:
        plt.ylim(ylim)

    if ymin:
        plt.ylim(bottom=ymin)

    plt.show()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")



def plot_metric_continue_evalu_V2(task, evalu_values_t1, evalu_values_t2, evalu_values_t3, evalu_values_t4, ylabel, title, save_path=None, ylim=None, ymin=None, ymax=None):
    plt.figure(figsize=(10, 5))
    evalu_values_t2_modified = evalu_values_t2.copy()
    evalu_values_t2_modified[0] = np.nan
    evalu_values_t3_modified = evalu_values_t3.copy()
    evalu_values_t3_modified[0] = np.nan
    evalu_values_t3_modified[1] = np.nan
    evalu_values_t4_modified = evalu_values_t4.copy()
    evalu_values_t4_modified[0] = np.nan
    evalu_values_t4_modified[1] = np.nan
    evalu_values_t4_modified[2] = np.nan
    plt.plot(task, evalu_values_t1, label=f"Task 4 {ylabel}", marker='o')
    plt.plot(task, evalu_values_t2_modified, label=f"Task 2 {ylabel}", marker='o')
    plt.plot(task, evalu_values_t3_modified, label=f"Task 3 {ylabel}", marker='o')
    plt.plot(task, evalu_values_t4_modified, label=f"Task 1 {ylabel}", marker='o')
    plt.xlabel("Version of Model")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.xticks(np.arange(1, 5, 1))

    if ylim:
        plt.ylim(ylim)

    if ymin:
        plt.ylim(bottom=ymin)

    plt.show()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
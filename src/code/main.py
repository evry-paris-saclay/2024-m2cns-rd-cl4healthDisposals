import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import Subset, DataLoader

from custom_dataset import CustomImageDataset
import random
import os
import pickle
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from plot import plot_silhouette_scores

from task2vec.aws_cv_task2vec.task2vec import Task2Vec
from task2vec.aws_cv_task2vec import task_similarity
from leep.leep import log_expected_empirical_prediction as leep

# Compute distance matrix
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

from model.model_resnet import resnet_model, resnet_model_leep, resnet_model_leep_val
from distance import generate_tasks, euclidean
from exp.exp_flatten import exp_flatten
from exp.exp_specific import exp_specific
from exp.exp_mtl import exp_mtl
from exp.exp_continual import exp_continual


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')  # Using this if in MacOS
BATCH_SIZE = 16

data_transform = transforms.Compose([
    transforms.Resize((40, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

data_dir = '/Users/jiaqifeng/Downloads/Medical Waste 4.0'
# dataset = datasets.ImageFolder(root=data_dir, transform=data_transform)
random.seed(42)

global_classes = [
    'glove_pair_latex', 'glove_pair_nitrile', 'glove_pair_surgery',
    'glove_single_latex', 'glove_single_nitrile', 'glove_single_surgery',
    'shoe_cover_pair', 'shoe_cover_single',
    'urine_bag', 'gauze', 'medical_cap', 'medical_glasses', 'test_tube'
]

tache1_classes = ['glove_pair_latex', 'glove_pair_nitrile', 'glove_pair_surgery',
                  'glove_single_latex', 'glove_single_nitrile', 'glove_single_surgery']
tache2_classes = ['shoe_cover_pair', 'shoe_cover_single']
tache3_classes = ['urine_bag', 'gauze', 'medical_cap', 'medical_glasses', 'test_tube']

global_label_mapping = {label: idx for idx, label in enumerate(global_classes)}
print("Global Label Mapping:", global_label_mapping)
custom_dataset = CustomImageDataset(data_dir=data_dir,class2idx=global_label_mapping,transform=data_transform)

# tache1_label_mapping = {label: global_label_mapping[label] for label in tache1_classes}
# tache2_label_mapping = {label: global_label_mapping[label] for label in tache2_classes}
# tache3_label_mapping = {label: global_label_mapping[label] for label in tache3_classes}
# print("Tache 1 Mapping:", tache1_label_mapping)
# print("Tache 2 Mapping:", tache2_label_mapping)
# print("Tache 3 Mapping:", tache3_label_mapping, "\n")


def sample_and_resize_images(data_dir, label_mapping, num_samples_per_class=2):
    selected_images = {}
    resized_images = []

    for class_name, class_idx in label_mapping.items():
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Class folder {class_name} does not exist.")
            continue

        image_files = [os.path.join(class_dir, img) for img in os.listdir(class_dir) if
                       img.endswith(('.png', '.jpg', '.jpeg'))]
        if len(image_files) < num_samples_per_class:
            print(f"Less than {num_samples_per_class} images for class {class_name}")
            continue

        selected_files = random.sample(image_files, num_samples_per_class)
        selected_images[class_name] = selected_files

        for img_path in selected_files:
            image = Image.open(img_path).convert("RGB")
            resized_image = data_transform(image)
            resized_images.append((resized_image, class_name))

    return selected_images, resized_images


def show_and_save_images(original_images, resized_images, output_dir="output_images"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for class_name, orig_paths in original_images.items():
        for i, orig_path in enumerate(orig_paths):
            orig_image = Image.open(orig_path).convert("RGB")
            plt.figure(figsize=(6, 3))
            plt.imshow(orig_image)
            plt.axis("off")
            plt.title(f"Original - {class_name} (Sample {i + 1})")

            save_path_orig = os.path.join(output_dir, f"{class_name}_original_{i + 1}.png")
            plt.savefig(save_path_orig, dpi=300)
            plt.show()
            print(f"Saved original image: {save_path_orig}")

            resized_image_tensor = [img_tensor for img_tensor, cname in resized_images if cname == class_name][i]
            img_np = resized_image_tensor.permute(1, 2, 0).numpy()
            img_np = (img_np * 0.5) + 0.5
            plt.figure(figsize=(6, 3))
            plt.imshow(img_np)
            plt.axis("off")
            plt.title(f"Resized - {class_name} (Sample {i + 1})")

            save_path_resized = os.path.join(output_dir, f"{class_name}_resized_{i + 1}.png")
            plt.savefig(save_path_resized, dpi=300)
            plt.show()
            print(f"Saved resized image: {save_path_resized}")


def function_task2vec(global_label_mapping, custom_dataset, device, BATCH_SIZE=BATCH_SIZE):
    # model = resnet_model(global_classes, custom_dataset, device, BATCH_SIZE=BATCH_SIZE)
    model = torch.load('checkpoint/best_model.pth', weights_only=False)
    model.eval()

    # 获取数据集中的类名及数据
    embeddings = []
    dataset_names = []
    dataset_list = []
    for class_name, class_id in global_label_mapping.items():
        print(f"Processing class: {class_name}")
        indices = custom_dataset.get_task_subset_indices([class_name])
        if len(indices) == 0:
            print(f"No samples found for class {class_name}")
            continue

        subset = Subset(custom_dataset, indices)
        dataset_list.append(subset)
        dataset_names.append(class_name)
        print(f"Class {class_name} has {len(subset)} samples.")

    # 绘制任务之间的距离矩阵
    for name, dataset in zip(dataset_names, dataset_list):
        print(f"Embedding {name}")
        probe_network = model.to(device)
        embedding = Task2Vec(probe_network, max_samples=1000, skip_layers=6).embed(dataset)
        print(
            f"Embedding for {name}: Hessian shape: {embedding.hessian.shape}, Scale shape: {embedding.scale.shape}")
        embeddings.append(embedding)
        print(f"embedding: {embedding}")
        print(f"embeddings: {embeddings}")
        print(f"Number of embeddings: {len(embeddings)}")

    # 计算 Hessian 矩阵的成对距离
    cond_distance_matrix = pdist([embedding.hessian for embedding in embeddings])

    if cond_distance_matrix.size == 0:
        raise ValueError("Distance matrix is empty. Check embeddings for validity.")
    print(f"Distance Matrix: {cond_distance_matrix}")

    task_similarity.plot_distance_matrix(embeddings, dataset_names)
    task_similarity.plot_similarity_matrix(embeddings, dataset_names)
    return embeddings


def hierarchical_clustering_with_silhouette(embeddings, max_clusters=13, metric_distance='cosine'):
    distance_matrix = task_similarity.pdist(embeddings, distance=metric_distance)
    print("distance_matrix", distance_matrix)
    print("distance_matrix", distance_matrix.shape)

    cond_distance_matrix = squareform(distance_matrix, checks=False)
    print("cond_distance_matrix", cond_distance_matrix)
    print("cond_distance_matrix", cond_distance_matrix.shape)

    linkage_matrix = linkage(cond_distance_matrix, method='complete', optimal_ordering=True)

    silhouette_scores = {}
    for k in range(2, max_clusters + 1):  # 聚类数从 2 到 max_clusters
        cluster_labels = fcluster(linkage_matrix, k, criterion='maxclust')  # 生成聚类标签
        # print(f"cluster_labels : {cluster_labels}")
        score = silhouette_score(distance_matrix, cluster_labels, metric=metric_distance)
        print(score)
        silhouette_scores[k] = score
    best_k = max(silhouette_scores, key=silhouette_scores.get)
    plot_silhouette_scores(silhouette_scores)
    return best_k, linkage_matrix


def tsne(k, linkage_matrix, embeddings):
    n_clusters = k
    cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')  # 生成聚类标签
    print(f"max {n_clusters} cluster_labels : {cluster_labels}")

    tasks = {f"task_{label}": [] for label in list(cluster_labels)}
    for class_name, label in zip(global_classes, list(cluster_labels)):
        tasks[f"task_{label}"].append(class_name)

    print(f"Tasks: {tasks}")

    embeddings_array = np.array([embedding.hessian.flatten() for embedding in embeddings])
    print(embeddings_array.shape)  # 输出 (n_samples, n_features)

    tsne = TSNE(n_components=2, perplexity=5, random_state=42)
    data_2d = tsne.fit_transform(embeddings_array)

    plt.figure(figsize=(8, 6))
    for cluster in range(1, n_clusters + 1):
        cluster_points = data_2d[cluster_labels == cluster]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {cluster}")

    plt.title("t-SNE Visualization of Clusters")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend()
    plt.grid(True)
    plt.show()
    return tasks


def function_leep(custom_dataset, tasks, save_path="results/leep_score.png"):
    task_names = list(tasks.keys())  # 获取任务名称

    leep_scores = {}
    for source_task in task_names:  # 外层循环：源任务
        # resnet_model_leep(tasks[source_task], source_task, custom_dataset, device, BATCH_SIZE=BATCH_SIZE)
        for target_task in task_names:  # 内层循环：目标任务
            if source_task != target_task:  # 确保源任务和目标任务不同
                print(f"Computing LEEP score from {source_task} to {target_task}...")

                prediction_cible, labels_cible = resnet_model_leep_val(tasks[target_task], source_task, custom_dataset,
                                                                       device, BATCH_SIZE)
                score = leep(prediction_cible, labels_cible)

                leep_scores[(source_task, target_task)] = score
                print(f"LEEP score from {source_task} to {target_task}: {score}\n")

    # 打印所有任务对的 LEEP 分数
    print("\nAll LEEP scores:")
    for task_pair, score in leep_scores.items():
        print(f"{task_pair[0]} -> {task_pair[1]}: {score}")

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


def main():
    # global_label_mapping = {label: idx for idx, label in enumerate(global_classes)}
    # print("Global Label Mapping:", global_label_mapping)
    # custom_dataset = CustomImageDataset(data_dir=data_dir, class2idx=global_label_mapping, transform=data_transform)
    # tache3_label_mapping = {label: global_label_mapping[label] for label in tache3_classes}

    # # euclid
    # euclidean(global_classes, custom_dataset, device, BATCH_SIZE=BATCH_SIZE)

    # # task2vec
    # embeddings = function_task2vec(tache3_label_mapping, custom_dataset, device, BATCH_SIZE=BATCH_SIZE)
    # embeddings = function_task2vec(global_label_mapping, custom_dataset, device, BATCH_SIZE=BATCH_SIZE)

    # # 保存 embeddings 到文件
    # with open("embeddings.pkl", "wb") as f:
    #     pickle.dump(embeddings, f)

    # with open("embeddings.pkl", "rb") as f:
    #     embeddings = pickle.load(f)

    # best_k, linkage_matrix = hierarchical_clustering_with_silhouette(embeddings)

    # # task2vec tsne
    # tasks = tsne(best_k, linkage_matrix, embeddings)

    # # leep
    tasks = {
        "Tache1": tache1_classes,
        "Tache2": tache2_classes,
        "Tache3": tache3_classes
    }

    # resnet_model_leep(tache1_classes, "Tache1", custom_dataset, device, BATCH_SIZE=BATCH_SIZE)
    # resnet_model_leep(tache2_classes, "Tache2", custom_dataset, device, BATCH_SIZE=BATCH_SIZE)
    # resnet_model_leep(tache3_classes, "Tache3", custom_dataset, device, BATCH_SIZE=BATCH_SIZE)

    # prediction_cible, labels_cible = resnet_model_leep_val(tache3_classes, custom_dataset, device, BATCH_SIZE=BATCH_SIZE)
    # score = leep(prediction_cible, labels_cible)
    # print(f"Leep score: {score}")

    function_leep(custom_dataset, tasks)

    # # save images
    # original_images, resized_images = sample_and_resize_images(data_dir, global_label_mapping)
    # show_and_save_images(original_images, resized_images, output_dir="output_images")

    # # experiences
    # exp_flatten(device, custom_dataset)
    # exp_specific(device, custom_dataset, tache1_classes, tache2_classes, tache3_classes, BATCH_SIZE=BATCH_SIZE)
    # exp_mtl(device, custom_dataset, tache1_classes, tache2_classes, tache3_classes, BATCH_SIZE=BATCH_SIZE)
    # exp_continual(device, custom_dataset, tache1_classes, tache2_classes, tache3_classes, BATCH_SIZE=BATCH_SIZE)
    # exp_continual(device, custom_dataset, tasks, BATCH_SIZE=BATCH_SIZE)


if __name__ == '__main__':
    main()
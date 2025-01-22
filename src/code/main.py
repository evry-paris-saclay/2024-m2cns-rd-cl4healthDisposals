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
from plot import plot_silhouette_scores, plot_leep_scores_ligne, plot_leep_scores_heatmap, plot_leep_scores_order

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
from exp.exp_continual_v2 import exp_continual_v2


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')  # Using this if in MacOS
BATCH_SIZE = 16

data_transform = transforms.Compose([
    transforms.Resize((400, 640)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# data_dir = '/Users/jiaqifeng/Downloads/Medical Waste 4.0'
data_dir = '/home/hungry_gould/projet/2024-m2cns-rd-cl4healthDisposals/data/Medical Waste 4.0'
# dataset = datasets.ImageFolder(root=data_dir, transform=data_transform)
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

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

# global_label_mapping = {label: idx for idx, label in enumerate(global_classes)}
# print("Global Label Mapping:", global_label_mapping)
# custom_dataset = CustomImageDataset(data_dir=data_dir,class2idx=global_label_mapping,transform=data_transform)

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
    model = torch.load('/home/hungry_gould/projet/2024-m2cns-rd-cl4healthDisposals/src/code/checkpoint/resnet34.pth', weights_only=False)
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


def hierarchical_clustering_with_silhouette(embeddings, max_clusters=12, metric_distance='cosine'):
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


def tsne(k, linkage_matrix, embeddings, save_path="/home/hungry_gould/projet/2024-m2cns-rd-cl4healthDisposals/src/results/tsne_clustering.jpg"):
    n_clusters = k
    cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')  # 生成聚类标签
    print(f"max {n_clusters} cluster_labels : {cluster_labels}")

    tasks = {f"tache{label}": [] for label in list(cluster_labels)}
    for class_name, label in zip(global_classes, list(cluster_labels)):
        tasks[f"tache{label}"].append(class_name)

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
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.grid(True)

    plt.show()
    return tasks


def function_leep(custom_dataset, tasks):
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

    plot_leep_scores_ligne(leep_scores)
    plot_leep_scores_heatmap(leep_scores,task_names)
    output = trouver_order(leep_scores)

    sequence = output.split(" -> ")
    ordered_tasks = {task: tasks[task] for task in sequence}

    print("Reordered Tasks:")
    for task, items in ordered_tasks.items():
        print(f"{task}: {items}")

    return ordered_tasks
    

def trouver_order(leep_scores):
    sorted_leep_scores = sorted(leep_scores.items(), key=lambda x: x[1], reverse=True)
    # 初始化链条
    chain = {}  # 存储链条，形式为 [(source, target, score)]
    used_tasks = set()  # 用于记录已使用的任务
    source_tasks= set()
    target_tasks=set()
    num=0
    # 从最高分的任务对开始构建链条
    for (source, target), score in sorted_leep_scores:
        #第一个
        if source not in used_tasks and target not in used_tasks:
            #chain.append((source, target, score))
            chain[(source, target)] = score
            used_tasks.add(source)
            used_tasks.add(target)
            source_tasks.add(source)
            target_tasks.add(target)
            num+=1


        if (source not in source_tasks) and (target not in target_tasks) and (source not in used_tasks or target not in used_tasks):
            #chain.append((source, target, score))
            chain[(source, target)] = score
            used_tasks.add(source)
            used_tasks.add(target)
            source_tasks.add(source)
            target_tasks.add(target)
            num+=1

        if num==3:
            break
    
    # 输出最终的链式路径
    print("Final Task Chain:")
    for (source, target), score in chain.items():
        print(f"{source} -> {target}: {score:.2e}")

    sequence = []

    # Extract all source and target nodes from the chain
    source_nodes = {source for source, target in chain.keys()}
    target_nodes = {target for source, target in chain.keys()}

    # Find the starting node (not a target in any pair)
    start_node = (source_nodes - target_nodes).pop()

    # Traverse the chain to build the sequence
    current_node = start_node
    while current_node:
        sequence.append(current_node)
        next_node = next((target for (source, target) in chain.keys() if source == current_node), None)
        if next_node is None:
            break
        current_node = next_node

    # Join the sequence to form the output string
    output = " -> ".join(sequence)
    print(output)

    plot_leep_scores_order(chain, output)
    return output

def main():
    global_label_mapping = {label: idx for idx, label in enumerate(global_classes)}
    print("Global Label Mapping:", global_label_mapping)
    custom_dataset = CustomImageDataset(data_dir=data_dir, class2idx=global_label_mapping, transform=data_transform)
    # tache3_label_mapping = {label: global_label_mapping[label] for label in tache3_classes}


    # # euclid
    #resnet_model(global_classes, custom_dataset, device, BATCH_SIZE)
    #euclidean(global_classes, custom_dataset, device, BATCH_SIZE=BATCH_SIZE)


    # # task2vec
    # embeddings = function_task2vec(tache3_label_mapping, custom_dataset, device, BATCH_SIZE=BATCH_SIZE)
    # embeddings = function_task2vec(global_label_mapping, custom_dataset, device, BATCH_SIZE=BATCH_SIZE)

    # # 保存 embeddings 到文件
    # with open("/home/hungry_gould/projet/2024-m2cns-rd-cl4healthDisposals/src/code/embeddings.pkl", "wb") as f:
    #     pickle.dump(embeddings, f)

    with open("/home/hungry_gould/projet/2024-m2cns-rd-cl4healthDisposals/src/code/embeddings.pkl", "rb") as f:
        embeddings = pickle.load(f)

    best_k, linkage_matrix = hierarchical_clustering_with_silhouette(embeddings)


    # # task2vec tsne
    best = 4
    tasks = tsne(best, linkage_matrix, embeddings)

    tasks_sorted = function_leep(custom_dataset, tasks)
    # print("tasks_sorted",tasks_sorted)
    # with open("/home/hungry_gould/projet/2024-m2cns-rd-cl4healthDisposals/src/code/task_dict.pkl", "wb") as file:
    #     pickle.dump(tasks_sorted, file)

    # with open("/home/hungry_gould/projet/2024-m2cns-rd-cl4healthDisposals/src/code/task_dict.pkl", "rb") as file:
    #     tasks_sorted = pickle.load(file)
    # print("tasks_sorted",tasks_sorted)


    # # save images
    # original_images, resized_images = sample_and_resize_images(data_dir, global_label_mapping)
    # show_and_save_images(original_images, resized_images, output_dir="output_images")


    # # experiences
    # exp_flatten(device, custom_dataset)
    # exp_specific(device, custom_dataset, tache1_classes, tache2_classes, tache3_classes, BATCH_SIZE=BATCH_SIZE)
    # exp_mtl(device, custom_dataset, tache1_classes, tache2_classes, tache3_classes, BATCH_SIZE=BATCH_SIZE)
    # exp_continual(device, custom_dataset, tache1_classes, tache2_classes, tache3_classes, BATCH_SIZE=BATCH_SIZE)
    # exp_continual_v2(device, custom_dataset, tasks_sorted, BATCH_SIZE=BATCH_SIZE)


if __name__ == '__main__':
    main()

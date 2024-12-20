import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics.pairwise import euclidean_distances

from model.model_resnet import resnet_model
from plot import determine_optimal_clusters_silhouette, determine_optimal_clusters


def extract_class_features(dataset, model, device):
    class_features = {class_name: [] for class_name in dataset.classes}

    model.eval()
    with torch.no_grad():
        for img, label in DataLoader(dataset, batch_size=32, shuffle=False):
            img = img.to(device)
            features = model(img).squeeze()
            for feat, lbl in zip(features, label):
                class_name = dataset.classes[lbl]
                class_features[class_name].append(feat.cpu().numpy())

    class_centroids = {class_name: np.mean(features, axis=0) for class_name, features in class_features.items()}
    return class_centroids


# Cluster classes by distance
def cluster_classes_by_distance(class_centroids, n_clusters=None):
    class_names = list(class_centroids.keys())
    centroids = np.array(list(class_centroids.values()))

    # Compute distance matrix
    distance_matrix = euclidean_distances(centroids, centroids)

    # Perform hierarchical clustering
    clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage='average')
    labels = clustering.fit_predict(distance_matrix)

    # Perform KMeans
    # clustering = KMeans(n_clusters=n_clusters, random_state=42)
    # labels = clustering.fit_predict(centroids)

    # Group classes into tasks
    tasks = {f"Task_{label}": [] for label in set(labels)}
    for class_name, label in zip(class_names, labels):
        tasks[f"Task_{label}"].append(class_name)

    return tasks


def generate_tasks(global_classes, custom_dataset, device, BATCH_SIZE):
    print("Training model from scratch...")
    model = resnet_model(global_classes, custom_dataset, device, BATCH_SIZE=BATCH_SIZE)

    print("Extracting class features...")
    feature_extractor = nn.Sequential(*list(model.children())[:-1])
    feature_extractor.to(device)
    class_centroids = extract_class_features(custom_dataset, feature_extractor, device)

    # Cluster classes into tasks
    print("Clustering classes...")
    tasks = cluster_classes_by_distance(class_centroids, n_clusters=2)
    print("Generated tasks:", tasks)

    # Output generated tasks
    print("Generated tasks and their classes:")
    for task_name, classes in tasks.items():
        print(f"{task_name}: {classes}")

    # Create task-specific loaders
    task_loaders = {}
    for task_name, classes in tasks.items():
        task_loaders[task_name] = custom_dataset.create_task_loaders(classes, batch_size=16)

    # Example: Iterate over a task loader
    # for task_name, (train_loader, val_loader) in task_loaders.items():
    #     print(f"Running experiments for {task_name}")
    #     for images, labels in train_loader:
    #         print(f"Task {task_name} - Batch of images: {images.size()}, Labels: {labels}")
    #         break

    determine_optimal_clusters_silhouette(class_centroids)
    # determine_optimal_clusters(class_centroids, max_clusters=10)
import numpy as np
import torch
from torch.utils.data import Subset, DataLoader

from custom_dataset import CustomImageDataset
import random
import os
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms, datasets

from task2vec.aws_cv_task2vec.task2vec import Task2Vec
from task2vec.aws_cv_task2vec import task_similarity
from leep.leep import log_expected_empirical_prediction as leep, plot_distance_matrix_leep

# Compute distance matrix
from scipy.spatial.distance import pdist, squareform

from model.model_resnet import resnet_model
from distance import generate_tasks
from exp.exp_flatten import exp_flatten
from exp.exp_specific import exp_specific
from exp.exp_mtl import exp_mtl
from exp.exp_continual import exp_continual
from sklearn.cluster import KMeans


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

# tache1_classes = ['glove_pair_latex', 'glove_pair_nitrile', 'glove_pair_surgery',
#                   'glove_single_latex', 'glove_single_nitrile', 'glove_single_surgery']
# tache2_classes = ['shoe_cover_pair', 'shoe_cover_single']
# tache3_classes = ['urine_bag', 'gauze', 'medical_cap', 'medical_glasses', 'test_tube']

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


def function_task2vec(global_classes, custom_dataset, device, BATCH_SIZE=BATCH_SIZE):
    # model = resnet_model(global_classes, custom_dataset, device, BATCH_SIZE=BATCH_SIZE)
    model = torch.load('checkpoint/best_model.pth', weights_only=False)
    model.eval()

    # num_tasks = 5
    # kmeans = KMeans(n_clusters=num_tasks, random_state=42)
    # labels = kmeans.fit_predict(task_vectors)
    #
    # for i, label in enumerate(labels):
    #     print(f"Class {global_classes[i]} is assigned to task {label}")

    # embedding = Task2Vec(model).embed(custom_dataset)

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
        print(f"Embedding for {name}: Hessian shape: {embedding.hessian.shape}, Scale shape: {embedding.scale.shape}")
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


def function_leep(global_classes, custom_dataset, device, BATCH_SIZE=BATCH_SIZE):
    model = torch.load('checkpoint/best_model.pth', weights_only=False).to(device)
    model.eval()

    dataset_names = []
    dataset_list = []
    leep_scores = []  # 存储每个类的 LEEP 分数

    # 处理每个类，创建数据子集
    for class_name, class_id in global_label_mapping.items():
        print(f"Processing class: {class_name}")
        indices = custom_dataset.get_task_subset_indices([class_name])
        if len(indices) == 0:
            print(f"No samples found for class {class_name}")
            continue

        # subset = Subset(custom_dataset, indices)
        # dataset_list.append(subset)
        loader_train = custom_dataset.create_subset_loaders([class_name], batch_size=BATCH_SIZE)
        dataset_names.append(class_name)
        print(f"Class {class_name} has {len(indices)} samples.")

        # 计算当前任务的 LEEP 分数
        predictions = []
        targets = []
        print(f"Calculating LEEP for {class_name}")
        with torch.no_grad():
            for inputs, labels in loader_train:  # 使用训练集加载器
                inputs = inputs.to(device)
                # print(f"Input shape: {inputs.shape}")  # 检查输入形状是否正确
                outputs = model(inputs)  # 获取模型预测
                predictions.append(outputs.cpu().numpy())
                targets.append(labels.cpu().numpy())

        # 转换为 numpy 数组
        predictions = np.vstack(predictions)
        targets = np.concatenate(targets)

        # 计算 LEEP 分数
        score = leep(predictions, targets)
        print(f"LEEP Score for {class_name}: {score}")
        leep_scores.append(score)

    # 如果需要进一步分析任务相似性，可以使用 LEEP 分数构建任务间距离矩阵
    print("Calculating pairwise distances between tasks...")
    cond_distance_matrix = pdist(np.array(leep_scores).reshape(-1, 1))  # 使用 LEEP 分数计算成对距离

    if cond_distance_matrix.size == 0:
        raise ValueError("Distance matrix is empty. Check LEEP scores for validity.")
    print(f"Distance Matrix: {cond_distance_matrix}")

    # 绘制任务之间的距离矩阵
    plot_distance_matrix_leep(leep_scores, dataset_names)


def main():
    # function_task2vec(global_classes, custom_dataset, device, BATCH_SIZE=BATCH_SIZE)
    function_leep(global_classes, custom_dataset, device, BATCH_SIZE=BATCH_SIZE)

    # generate_tasks(global_classes, custom_dataset, device, BATCH_SIZE=BATCH_SIZE)

    # original_images, resized_images = sample_and_resize_images(data_dir, global_label_mapping)
    # show_and_save_images(original_images, resized_images, output_dir="output_images")

    # exp_flatten(device, custom_dataset)
    # exp_specific(device, custom_dataset, tache1_classes, tache2_classes, tache3_classes, BATCH_SIZE=BATCH_SIZE)
    # exp_mtl(device, custom_dataset, tache1_classes, tache2_classes, tache3_classes, BATCH_SIZE=BATCH_SIZE)
    # exp_continual(device, custom_dataset, tache1_classes, tache2_classes, tache3_classes, BATCH_SIZE=BATCH_SIZE)


if __name__ == '__main__':
    main()
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset


def compute_fisher_information(model, global_classes, custom_dataset, device):
    task_vectors = []
    model = nn.Sequential(*list(model.children())[:-1])

    for class_idx in range(len(global_classes)):
        print(f"Processing class {class_idx + 1}/{len(global_classes)}: {global_classes[class_idx]}")

        indices = [i for i, (_, label) in enumerate(custom_dataset) if label == class_idx]
        class_subset = Subset(custom_dataset, indices)
        dataloader = DataLoader(class_subset, batch_size=16, shuffle=False)

        model.eval()
        fisher_information = torch.zeros_like(torch.cat([p.flatten() for p in model.parameters()]))

        for inputs, _ in dataloader:
            inputs = inputs.to(device)

            model.zero_grad()
            features = model(inputs)
            loss = torch.sum(features)
            loss.backward()

            gradients = torch.cat([p.grad.flatten() for p in model.parameters()])
            fisher_information += gradients ** 2

        fisher_information /= len(dataloader.dataset)
        task_vectors.append(fisher_information.cpu().numpy())

    task_vectors = np.array(task_vectors)
    print("Task2Vec vector matrix size:", task_vectors.shape)
    return task_vectors


def task2vec_embedding(fisher_information):
    """
    生成任务嵌入向量
    :param fisher_information: Fisher 信息矩阵的对角元素
    :param model: 预训练模型
    :return: 任务嵌入向量
    """
    embedding = fisher_information.mean(dim=0)
    return embedding

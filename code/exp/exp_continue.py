import numpy as np
import random
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset, ConcatDataset
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim
from collections import Counter

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, roc_auc_score

from model.model_continue import ContinueModel
from custom_dataset import CustomImageDataset
from plot import plot_metrics

replay_buffer = []
BUFFER_SIZE = 500  # 缓冲区最大容量
BUFFER_SAMPLE_SIZE = 64  # 每次从缓冲区采样的大小


def add_to_replay_buffer(inputs, labels):
    global replay_buffer

    # 确保 inputs 和 labels 是张量并转到 CPU
    inputs, labels = inputs.detach().cpu(), labels.detach().cpu()

    # 遍历 Batch 中的每个样本，逐个加入缓冲区
    for i in range(inputs.size(0)):
        if len(replay_buffer) >= BUFFER_SIZE:
            replay_buffer.pop(0)  # 如果缓冲区已满，移除最早的样本
        replay_buffer.append((inputs[i], labels[i]))


def sample_from_replay_buffer():
    global replay_buffer
    if len(replay_buffer) == 0:
        return None, None

    # 随机采样 BUFFER_SAMPLE_SIZE 个样本
    sampled = random.sample(replay_buffer, min(len(replay_buffer), BUFFER_SAMPLE_SIZE))
    sampled_inputs, sampled_labels = zip(*sampled)
    return torch.stack(sampled_inputs), torch.tensor(sampled_labels)


def exp_continue(device, custom_dataset, tache1_classes, tache2_classes, tache3_classes, tache4_classes, BATCH_SIZE):
    tache1_loader_train, tache1_loader_val = custom_dataset.create_task_loaders(tache1_classes, batch_size=BATCH_SIZE)
    tache2_loader_train, tache2_loader_val = custom_dataset.create_task_loaders(tache2_classes, batch_size=BATCH_SIZE)
    tache3_loader_train, tache3_loader_val = custom_dataset.create_task_loaders(tache3_classes, batch_size=BATCH_SIZE)
    tache4_loader_train, tache4_loader_val = custom_dataset.create_task_loaders(tache4_classes, batch_size=BATCH_SIZE)

    train_loaders = [tache1_loader_train, tache2_loader_train, tache3_loader_train, tache4_loader_train]
    val_loaders = [tache1_loader_val, tache2_loader_val, tache3_loader_val, tache4_loader_val]

    learning_rate = 0.001
    num_epochs = 5
    min_val_loss = float('inf')

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    all_labels = []
    all_probs = []

    # 实例化模型
    model = ContinueModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    cumulative_val_loader = None

    # 依次遍历每个任务的数据加载器
    for task_idx, train_loader in enumerate(train_loaders):
        print(f"\n=== Task {task_idx + 1} ===")

        if cumulative_val_loader is None:
            cumulative_val_loader = val_loaders[task_idx]
        else:
            combined_dataset = ConcatDataset([cumulative_val_loader.dataset, val_loaders[task_idx].dataset])
            cumulative_val_loader = DataLoader(combined_dataset, batch_size=BATCH_SIZE, shuffle=False)

        for epoch in range(num_epochs):
            print(f"\n--- Epoch {epoch + 1}/5 ---")

            model.train()
            total_loss, total_acc = 0, 0
            total_samples = 0

            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 从缓冲区采样
                replay_inputs, replay_labels = sample_from_replay_buffer()
                # print("replay_inputs:\n",replay_inputs)
                # print("replay_labels:\n",replay_labels)

                # 如果缓冲区非空，拼接缓冲区样本与当前任务样本
                if replay_inputs is not None:
                    replay_inputs = replay_inputs.to(device)
                    replay_labels = replay_labels.to(device)
                    inputs = torch.cat([inputs, replay_inputs])
                    labels = torch.cat([labels, replay_labels])

                # 模型前向传播和优化
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # 统计损失和准确率
                total_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                total_acc += torch.sum(preds == labels).item()
                total_samples += labels.size(0)

                # 加入缓冲区
                add_to_replay_buffer(inputs.detach().cpu(), labels.detach().cpu())

            # 计算平均损失和准确率
            avg_loss = total_loss / total_samples
            avg_accuracy = total_acc / total_samples
            train_losses.append(avg_loss)
            train_accuracies.append(avg_accuracy)
            print(f"Train Loss: {avg_loss:.4f}, Train Accuracy: {avg_accuracy:.4f}")

            model.eval()
            val_total_loss, val_total_acc, val_total_samples = 0, 0, 0

            with torch.no_grad():
                for val_inputs, val_labels in cumulative_val_loader:
                    val_inputs = val_inputs.to(device)
                    val_labels = val_labels.to(device)

                    # 前向传播
                    val_outputs = model(val_inputs)
                    val_loss = criterion(val_outputs, val_labels)
                    val_total_loss += val_loss.item() * val_inputs.size(0)

                    # 计算准确率
                    _, val_preds = torch.max(val_outputs, 1)
                    val_total_acc += torch.sum(val_preds == val_labels).item()
                    val_total_samples += val_labels.size(0)

            # 计算平均验证损失和准确率
            avg_val_loss = val_total_loss / val_total_samples
            avg_val_accuracy = val_total_acc / val_total_samples
            val_losses.append(avg_val_loss)
            val_accuracies.append(avg_val_accuracy)
            print(f"Val Loss: {avg_val_loss:.4f}, Val Accuracy: {avg_val_accuracy:.4f}")

            # 保存最优模型
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                # torch.save(model.state_dict(), f"/Users/jiaqifeng/PycharmProjects/Python_RD/checkpoint/best_model_task{task_idx}.pth")
                print(f"Model for Task {task_idx + 1} saved.")

    print("\n=== Final Metrics ===")
    print("Train Losses:", train_losses)
    print("Train Accuracies:", train_accuracies)
    print("Validation Losses:", val_losses)
    print("Validation Accuracies:", val_accuracies)

    print("Experiment 3 Finished!")
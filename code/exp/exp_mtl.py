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

from model.model_mtl import TotalModel, backbone, classifier
from custom_dataset import CustomImageDataset
from plot import plot_metrics

import matplotlib
import matplotlib.pyplot as plt

# matplotlib.use('TkAgg')
matplotlib.use('MacOSX')  # Using this if in MacOS


def exp_mtl(device, custom_dataset, tache1_classes, tache2_classes, tache3_classes, BATCH_SIZE):
    tache1_loader_train, tache1_loader_val = custom_dataset.create_task_loaders(tache1_classes, batch_size=23)
    tache2_loader_train, tache2_loader_val = custom_dataset.create_task_loaders(tache2_classes, batch_size=8)
    tache3_loader_train, tache3_loader_val = custom_dataset.create_task_loaders(tache3_classes, batch_size=20)

    class_input_dim = 8 * (64 - 4) * (40 - 4)  # 根据卷积层计算
    learning_rate = 0.001
    num_epochs = 30

    # best_model_path = '/Users/jiaqifeng/PycharmProjects/Python_RD/checkpoint/best_model.pth'
    min_val_loss = float('inf')

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    all_labels = []
    all_probs = []

    # 实例化模型
    model = TotalModel(class_input_dim=class_input_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练和验证循环
    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")
        model.train()

        total_loss = 0
        total_acc = 0
        total_batches = 0

        task_loaders = [tache1_loader_train, tache2_loader_train, tache3_loader_train]
        task_iters = [iter(loader) for loader in task_loaders]

        for iteration in range(67):
            loss = 0

            # 遍历每个任务的迭代器，获取一个 batch
            batch_data = [next(task_iter) for task_iter in task_iters]
            optimizer.zero_grad()

            for task_idx, (inputs, labels) in enumerate(batch_data):
                inputs = inputs.to(device)
                labels = labels.to(device)

                if task_idx == 1:  # Task 2
                    labels = labels - 6
                elif task_idx == 2:  # Task 3
                    labels = labels - 8

                outputs = model(inputs, task_idx)

                loss += criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                acc = (preds == labels).sum().item() / labels.size(0)
                total_acc += acc
                total_batches += 1

            loss.backward()
            optimizer.step()

            total_loss += loss

        avg_train_loss = (total_loss).item() / total_batches
        avg_train_acc = total_acc / total_batches * 100
        train_losses.append(avg_train_loss)
        train_accuracies.append(avg_train_acc)
        print(f"Train Loss: {avg_train_loss:.4f}, "
              f"Train Accuracy: {avg_train_acc:.2f}%")

        # 验证阶段
        model.eval()
        val_loss, val_acc = 0, 0
        val_batches = 0

        epoch_labels = []
        epoch_probs = []

        task_loaders_val = [tache1_loader_val, tache2_loader_val, tache3_loader_val]
        task_iters_val = [iter(loader) for loader in task_loaders]

        with torch.no_grad():
            for iteration in range(17):
                batch_data_val = [next(task_iter_val) for task_iter_val in task_iters_val]

                for task_idx, (val_inputs, val_labels) in enumerate(batch_data_val):
                    val_inputs = val_inputs.to(device)
                    val_labels = val_labels.to(device)

                    if task_idx == 1:  # Task 2
                        val_labels = val_labels - 6
                    elif task_idx == 2:  # Task 3
                        val_labels = val_labels - 8

                    val_outputs = model(val_inputs, task_idx)
                    val_loss += criterion(val_outputs, val_labels).item()

                    _, val_preds = torch.max(val_outputs, 1)
                    val_acc += (val_preds == val_labels).sum().item() / val_labels.size(0)
                    val_batches += 1

        avg_val_loss = val_loss / val_batches
        avg_val_acc = val_acc / val_batches * 100
        val_losses.append(avg_val_loss)
        val_accuracies.append(avg_val_acc)
        print(f"Validation Loss: {avg_val_loss:.4f}, "
              f"Validation Accuracy: {avg_val_acc:.2f}%")

        if avg_val_loss < min_val_loss:
            min_val_loss = avg_val_loss
            # torch.save(model.state_dict(), best_model_path)
            print(f"Model saved with Validation Loss: {min_val_loss:.4f}")

    # 绘制训练和验证曲线
    epochs = range(1, num_epochs + 1)

    # 绘制损失曲线
    plot_metrics(
        epochs=epochs,
        train_values=train_losses,
        val_values=val_losses,
        ylabel="Loss",
        title="Training and Validation Loss"
    )

    # 绘制准确率曲线
    plot_metrics(
        epochs=epochs,
        train_values=train_accuracies,
        val_values=val_accuracies,
        ylabel="Accuracy (%)",
        title="Training and Validation Accuracy"
    )

    print("Experiment 2 Finished !")
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

def exp_mtl(device, custom_dataset, tache1_classes, tache2_classes, tache3_classes, tache4_classes, BATCH_SIZE):
    tache1_loader_train, tache1_loader_val = custom_dataset.create_task_loaders(tache1_classes, batch_size=BATCH_SIZE)
    tache2_loader_train, tache2_loader_val = custom_dataset.create_task_loaders(tache2_classes, batch_size=BATCH_SIZE)
    tache3_loader_train, tache3_loader_val = custom_dataset.create_task_loaders(tache3_classes, batch_size=BATCH_SIZE)
    tache4_loader_train, tache4_loader_val = custom_dataset.create_task_loaders(tache4_classes, batch_size=BATCH_SIZE)

    class_input_dim = 8 * (32 - 4) * (32 - 4)  # 根据卷积层计算
    num_classes = 13
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
    model = TotalModel(class_input_dim=class_input_dim, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练和验证循环
    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")
        model.train()
        total_loss, total_acc = 0, 0
        total_batches = 0

        # 依次遍历每个任务的数据加载器
        for task_idx, loader in enumerate([tache1_loader_train, tache2_loader_train, tache3_loader_train, tache4_loader_train]):
            for inputs, labels in loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # 前向传播，指定任务索引
                outputs = model(inputs, task_idx)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # 计算准确率
                _, preds = torch.max(outputs, 1)
                acc = (preds == labels).sum().item() / labels.size(0)
                total_loss += loss.item()
                total_acc += acc
                total_batches += 1

        avg_train_loss = total_loss / total_batches
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

        with torch.no_grad():
            for task_idx, loader in enumerate([tache1_loader_val, tache2_loader_val, tache3_loader_val, tache4_loader_val]):
                for val_inputs, val_labels in loader:
                    val_inputs = val_inputs.to(device)
                    val_labels = val_labels.to(device)
                    val_outputs = model(val_inputs, task_idx)

                    # 获取 softmax 概率
                    val_probs = torch.softmax(val_outputs, dim=1).cpu().numpy()
                    epoch_probs.append(val_probs)
                    epoch_labels.append(val_labels.cpu().numpy())

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

        # 收集 ROC 数据
        all_labels.extend(np.concatenate(epoch_labels, axis=0))
        all_probs.extend(np.concatenate(epoch_probs, axis=0))

    # 转换为 NumPy 数组
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # 转换为 One-Hot 编码
    all_labels_onehot = label_binarize(all_labels, classes=range(num_classes))

    # 逐类别绘制 ROC 曲线
    plt.figure(figsize=(10, 8))
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(all_labels_onehot[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"Class {i} (AUC = {roc_auc:.2f})")

    # 计算宏平均和微平均 AUC
    macro_auc = roc_auc_score(all_labels_onehot, all_probs, average="macro", multi_class="ovr")
    micro_auc = roc_auc_score(all_labels_onehot, all_probs, average="micro", multi_class="ovr")

    # ROC 曲线
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve for Each Class\nMacro-AUC: {macro_auc:.2f}, Micro-AUC: {micro_auc:.2f}")
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()

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

    print("Experiment 2 Finished!")
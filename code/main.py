import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim
from collections import Counter

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, roc_auc_score

from model import ConvModel
from model_mlt import TotalModel, backbone, classifier
from custom_dataset import CustomImageDataset

import matplotlib
import matplotlib.pyplot as plt

# matplotlib.use('TkAgg')
matplotlib.use('MacOSX')  # Using this if in MacOS

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('mpa' if torch.cuda.is_available() else 'cpu')  # Using this if in MacOS
BATCH_SIZE = 16

data_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

data_dir = '/Users/jiaqifeng/Downloads/Medical Waste 4.0'
# dataset = datasets.ImageFolder(root=data_dir, transform=data_transform)
custom_dataset = CustomImageDataset(data_dir=data_dir, transform=data_transform)

tache1_classes = ['glove_pair_latex', 'glove_pair_nitrile', 'glove_pair_surgery',
                  'glove_single_latex', 'glove_single_nitrile', 'glove_single_surgery']
tache2_classes = ['shoe_cover_pair', 'shoe_cover_single']
tache3_classes = ['urine_bag']
tache4_classes = ['gauze', 'medical_cap', 'medical_glasses', 'test_tube']


def plot_metrics(epochs, train_values, val_values, ylabel, title, train_label="Training", val_label="Validation", save_path=None):
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_values, label=f"{train_label} {ylabel}", marker='o')
    plt.plot(epochs, val_values, label=f"{val_label} {ylabel}", marker='o')
    plt.xlabel("Epochs")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def exp_flatten():
    training_loader, validation_loader = custom_dataset.create_task_loaders(custom_dataset.dataset.classes)
    model = ConvModel().to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    for epoch in range(30):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for inputs, labels in training_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = loss_fn(outputs, labels)
            loss.backward()

            optimizer.step()
            running_loss += loss.item()

            _, predicted_labels = torch.max(outputs, 1)
            correct_predictions += (predicted_labels == labels).sum().item()
            total_predictions += labels.size(0)

        train_loss = running_loss / len(training_loader)
        train_losses.append(train_loss)
        train_accuracy = correct_predictions / total_predictions * 100
        train_accuracies.append(train_accuracy)
        print(f'Epoch [{epoch + 1}], Training Loss: {running_loss / len(training_loader):.4f}, Training Accuracy: {train_accuracy:.2f}%')

        model.eval()
        val_running_loss = 0.0
        val_correct_predictions = 0
        val_total_predictions = 0

        with torch.no_grad():
            for inputs, labels in validation_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)

                loss = loss_fn(outputs, labels)
                val_running_loss += loss.item()

                _, predicted_labels = torch.max(outputs, 1)
                val_correct_predictions += (predicted_labels == labels).sum().item()
                val_total_predictions += labels.size(0)

        val_loss = val_running_loss / len(validation_loader)
        val_losses.append(val_loss)
        val_accuracy = val_correct_predictions / val_total_predictions * 100
        val_accuracies.append(val_accuracy)
        print(f'Epoch [{epoch + 1}], Validation Loss: {val_running_loss / len(validation_loader):.4f}, Validation Accuracy: {val_accuracy:.2f}%')

    # 绘制并保存损失曲线
    epochs = range(1, len(train_losses) + 1)
    plot_metrics(
        epochs=epochs,
        train_values=train_losses,
        val_values=val_losses,
        ylabel="Loss",
        title="Training and Validation Loss",
        train_label="Training",
        val_label="Validation"
        # save_path="training_validation_loss.png"
    )

    # 绘制并保存准确率曲线
    epochs = range(1, len(val_losses) + 1)
    plot_metrics(
        epochs=epochs,
        train_values=train_accuracies,
        val_values=val_accuracies,
        ylabel="Accuracy",
        title="Training and Validation Accuracy",
        train_label="Training",
        val_label="Validation"
        # save_path="training_validation_accuracy.png"
    )

    print("Experiment 1 Finish !")


def exp_mlt():
    tache1_loader_train, tache1_loader_val = custom_dataset.create_task_loaders(tache1_classes, batch_size=BATCH_SIZE)
    tache2_loader_train, tache2_loader_val = custom_dataset.create_task_loaders(tache2_classes, batch_size=BATCH_SIZE)
    tache3_loader_train, tache3_loader_val = custom_dataset.create_task_loaders(tache3_classes, batch_size=BATCH_SIZE)
    tache4_loader_train, tache4_loader_val = custom_dataset.create_task_loaders(tache4_classes, batch_size=BATCH_SIZE)

    input_dim = 8 * (32 - 4) * (32 - 4)  # 根据卷积层计算
    num_classes = 13
    learning_rate = 0.001
    num_epochs = 30

    # best_model_path = 'best_model.pth'
    min_val_loss = float('inf')

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    all_labels = []
    all_probs = []

    # 实例化模型
    model = TotalModel(input_dim=input_dim, num_classes=num_classes).to(device)
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


def main():
#     exp_flatten()
    exp_mlt()


if __name__ == '__main__':
    main()
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from model import ConvModel

# 检查是否有可用的 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 16

# 定义图像转换，包括Resize和Normalize
transform = transforms.Compose([
    transforms.Resize((28, 28)),  # 调整图像大小
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

data_dir = '/Users/jiaqifeng/Downloads/Medical Waste 4.0'
dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# 使用 DataLoader 进行批次加载
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# 打印数据集类别
print(f'Classes: {dataset.classes}')


def main():
    # 初始化模型并将其移动到设备上
    model = ConvModel().to(device)

    # 定义损失函数和优化器
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()  # 将模型设置为训练模式

    # 训练过程
    for epoch in range(5):  # 假设训练5个epoch
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for inputs, labels in dataloader:
            # 将输入数据和标签移动到设备
            inputs, labels = inputs.to(device), labels.to(device)

            # 梯度清零
            optimizer.zero_grad()

            # 前向传播
            outputs = model(inputs)

            # 计算损失
            loss = loss_fn(outputs, labels)

            # 反向传播
            loss.backward()

            # 更新参数
            optimizer.step()

            # 累加损失
            running_loss += loss.item()

            # 计算准确率
            _, predicted_labels = torch.max(outputs, 1)
            correct_predictions += (predicted_labels == labels).sum().item()
            total_predictions += labels.size(0)

        # 打印每个 epoch 的损失和准确率
        accuracy = correct_predictions / total_predictions * 100
        print(f'Epoch [{epoch + 1}], Loss: {running_loss / len(dataloader):.4f}, Accuracy: {accuracy:.2f}%')

    print("Finish！")


if __name__ == '__main__':
    main()

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim
from collections import Counter

class CustomImageDataset(Dataset):
    def __init__(self,data_dir,transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.dataset = datasets.ImageFolder(root=self.data_dir, transform=self.transform)

    def get_task_subset_indices(self, classes):
        """
        获取属于指定任务的样本索引
        :param classes: 当前任务的类别列表
        :return: 样本索引列表
        """
        indices = []
        for idx, (_, label) in enumerate(self.dataset):
            class_name = self.dataset.classes[label]  # 获取类别名称
            if class_name in classes:
                indices.append(idx)
        return indices

    def create_task_loaders(self, classes, batch_size=32, train_split=0.8, shuffle=True):
        """
        创建特定任务的 DataLoader
        :param classes: 当前任务的类别列表
        :param batch_size: 每批次加载的样本数量
        :param train_split: 训练集占比
        :param shuffle: 是否打乱数据
        :return: (训练集 DataLoader, 验证集 DataLoader)
        """
        # 获取当前任务的样本索引
        indices = self.get_task_subset_indices(classes)
        task_subset = Subset(self.dataset, indices)

        # 按比例划分训练集和验证集
        train_size = int(train_split * len(task_subset))
        valid_size = len(task_subset) - train_size
        training_set, validation_set = random_split(task_subset, [train_size, valid_size])

        # 创建 DataLoader
        training_loader = DataLoader(training_set, batch_size=batch_size, shuffle=shuffle)
        validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False)

        print(f'Training set has {len(training_set)} instances')
        print(f'Validation set has {len(validation_set)} instances')
        print(f'Classes in Subset: {classes}\n')
        return training_loader, validation_loader
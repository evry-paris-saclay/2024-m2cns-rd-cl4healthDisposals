import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim
from collections import Counter


class CustomImageDataset(Dataset):
    def __init__(self,data_dir,class2idx,transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.dataset = datasets.ImageFolder(root=self.data_dir, transform=self.transform)
        self.class2idx = class2idx
        self.classes = list(class2idx.keys())

    def __len__(self):
        return len(self.dataset)

    # Renumber the labels in the underlying dataset to suit the needs of the task
    # be called implicitly by DataLoader
    def __getitem__(self,index):
        img,original_label = self.dataset[index]
        original_class = self.dataset.classes[original_label]
        mapped_label = self.class2idx[original_class]
        return img, mapped_label

    # Creating a dedicated data subset for each task in multi-task learning
    def get_task_subset_indices(self, classes):
        indices = []
        class_indices = [self.dataset.class_to_idx[cls_name] for cls_name in classes]
        for idx, label in enumerate(self.dataset.targets):
            if label in class_indices:
                indices.append(idx)
        return indices

    def create_task_loaders(self, classes, batch_size, train_split=0.8, shuffle=True):
        indices = self.get_task_subset_indices(classes)
        task_subset = Subset(self, indices)

        train_size = int(train_split * len(task_subset))
        valid_size = len(task_subset) - train_size
        training_set, validation_set = random_split(task_subset, [train_size, valid_size])

        training_loader = DataLoader(training_set, batch_size=batch_size, shuffle=shuffle)
        validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False)

        print(f'Training set has {len(training_set)} instances')
        print(f'Validation set has {len(validation_set)} instances')
        print(f'Classes in Subset: {classes}\n')
        return training_loader, validation_loader

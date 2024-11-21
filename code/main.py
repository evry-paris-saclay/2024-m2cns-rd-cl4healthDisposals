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

from model.model_flatten import ConvModel
from model.model_mtl import TotalModel, backbone, classifier
from model.model_continue import ContinueModel

from custom_dataset import CustomImageDataset
from plot import plot_metrics

from exp.exp_flatten import exp_flatten
from exp.exp_mtl import exp_mtl
from exp.exp_continue import exp_continue

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


def main():
    exp_flatten(device, custom_dataset)
#     exp_mtl(device, custom_dataset, tache1_classes, tache2_classes, tache3_classes, tache4_classes, BATCH_SIZE=BATCH_SIZE)
#     exp_continue(device, custom_dataset, tache1_classes, tache2_classes, tache3_classes, tache4_classes, BATCH_SIZE=BATCH_SIZE)


if __name__ == '__main__':
    main()
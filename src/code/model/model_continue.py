import torch.nn as nn

BATCH_SIZE = 16
CHANNELS_IN = 1
HEIGHT = 64-4
WIDTH = 40-4


class backbone(nn.Module):
    def __init__(self):
        super(backbone, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 8, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        return x


class classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(classifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = self.fc(x)
        return x


class ContinueModel(nn.Module):
    def __init__(self,class_input_dim):
        super(ContinueModel, self).__init__()
        self.backbones = backbone()
        self.head_tache1 = classifier(class_input_dim,6)
        self.head_tache2 = classifier(class_input_dim,2)
        self.head_tache3 = classifier(class_input_dim,5)

    def forward(self,x,task_idx):
        feature = self.backbones(x)
        if task_idx == 0:
            output = self.head_tache1(feature)
        elif task_idx == 1:
            output = self.head_tache2(feature)
        else:
            output = self.head_tache3(feature)
        return output

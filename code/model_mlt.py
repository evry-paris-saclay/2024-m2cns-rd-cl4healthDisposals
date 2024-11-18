import torch.nn as nn


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

# create main model
class TotalModel(nn.Module):
    def __init__(self,input_dim,num_classes):
        super(TotalModel,self).__init__()
        self.backbones = nn.ModuleList([backbone() for _ in range(4)])
        self.classifier = classifier(input_dim,num_classes)

    def forward(self,x,task_idx):
        """
        params: 4个任务数据
        result: 分类结果
        """
        feature = self.backbones[task_idx](x)
        output = self.classifier(feature)
        return output

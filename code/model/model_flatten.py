import torch.nn as nn

BATCH_SIZE = 16
CHANNELS_IN = 1
HEIGHT = 28
WIDTH = 28


class ConvModel(nn.Module):
    def __init__(self):
        super(ConvModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.fc = nn.Linear(32 * 28 * 28, 13)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc(x)
        return x

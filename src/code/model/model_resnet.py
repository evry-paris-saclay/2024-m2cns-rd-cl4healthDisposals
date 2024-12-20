import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models


def resnet_model(global_classes, custom_dataset, device, BATCH_SIZE):
    # Load feature extractor model
    # model = models.resnet50(weights='IMAGENET1K_V1')

    # Load full ResNet-50 model without pretrained weights
    model = models.resnet50(weights=None)
    num_classes = len(global_classes)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(device)

    # Optimizer and loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 1
    train_loader = DataLoader(custom_dataset, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

    return model
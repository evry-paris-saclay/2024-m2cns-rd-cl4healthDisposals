import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F
from torchvision import models
from task2vec.aws_cv_task2vec.models import resnet34

map_table = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 0,
    7: 1,
    8: 0,
    9: 1,
    10: 2,
    11: 3,
    12: 4,
}


def resnet_model(global_classes, custom_dataset, device, BATCH_SIZE):
    min_loss = float('inf')
    # Load feature extractor model
    # model = models.resnet50(weights='IMAGENET1K_V1')

    # Load full ResNet-34 model without pretrained weights
    # model = models.resnet50(weights=None)
    num_classes = len(global_classes)
    model = resnet34(num_classes=num_classes)
    # model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(device)

    # Optimizer and loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 10
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

        avg_loss = running_loss / len(train_loader)
        if avg_loss < min_loss:
            min_loss = avg_loss
            torch.save(model, f"/Users/jiaqifeng/PycharmProjects/Python_RD/checkpoint/best_model.pth")
            print(f"Model saved with Loss: {min_loss:.4f}")
    return model


def resnet_model_leep(taches_classes, custom_dataset, device, BATCH_SIZE):
    min_loss = float('inf')
    tache_loader = custom_dataset.create_subset_loaders(taches_classes, batch_size=BATCH_SIZE)

    num_classes = len(taches_classes)
    model = resnet34(num_classes=num_classes)
    # model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(device)

    # Optimizer and loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0
        total_loss, total_acc = 0, 0
        total_batches = 0
        for images, labels in tache_loader:
            mapped_labels = labels.clone()
            for old_label, new_label in map_table.items():
                mapped_labels[mapped_labels == old_label] = new_label

            images, labels = images.to(device), mapped_labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # running_loss += loss.item()
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            total_acc += (preds == labels).sum().item() / labels.size(0)
            total_batches += 1
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / total_batches}")

        avg_loss = total_loss / total_batches
        avg_accuracy = total_acc / total_batches * 100
        if avg_loss < min_loss:
            min_loss = avg_loss
            torch.save(model, f"checkpoint/resnet_tache1.pth")
            # torch.save(model, f"checkpoint/resnet_tache2.pth")
            # torch.save(model, f"checkpoint/resnet_tache3.pth")
            print(f"Model saved with Loss: {min_loss:.4f}")
        print(f"Train Loss: {avg_loss:.4f}, Train Accuracy: {avg_accuracy:.2f}%")


def resnet_model_leep_val(taches_classes_cible, target_task, custom_dataset, device, BATCH_SIZE):
    if target_task == "Tache1":
        model = torch.load('checkpoint/resnet_tache1.pth', weights_only=False)
    elif target_task == "Tache2":
        model = torch.load('checkpoint/resnet_tache2.pth', weights_only=False)
    model.eval()
    tache_cible_loader = custom_dataset.create_subset_loaders(taches_classes_cible, batch_size=BATCH_SIZE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    val_total_loss, val_total_acc, val_batches = 0, 0, 0
    labels_cible = []
    prediction_cible = []
    with torch.no_grad():
        for val_inputs, val_labels in tache_cible_loader:
            inputs = val_inputs.to(device)
            labels = val_labels.to(device)
            labels_cible.append(list(labels.cpu().numpy()))

            val_outputs = model(inputs)
            val_loss = criterion(val_outputs, labels)
            _ , preds = torch.max(val_outputs, 1)
            prediction_cible.append(list(val_outputs.cpu().numpy()))

            val_acc = (preds == labels).sum().item() / labels.size(0)
            val_total_loss += val_loss.item()
            val_total_acc += val_acc
            val_batches += 1

        avg_val_loss = val_total_loss / val_batches
        avg_val_acc = val_total_acc / val_batches * 100
        print(f"Val Loss: {avg_val_loss:.4f}, Val Accuracy: {avg_val_acc:.2f}%")

    prediction_cible = np.vstack(prediction_cible)
    labels_cible = np.concatenate(labels_cible)
    return prediction_cible, labels_cible

import os

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
    2: 0,
    3: 0,
    4: 2,
    5: 1,
    6: 2,
    7: 3,
    8: 3,
    9: 0,
    10: 4,
    11: 1,
    12: 2,
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
            torch.save(model, f"/home/hungry_gould/projet/2024-m2cns-rd-cl4healthDisposals/src/code/checkpoint/resnet34.pth")
            print(f"Model saved with Loss: {min_loss:.4f}")
    


def resnet_model_leep(taches_classes, source_task, custom_dataset, device, BATCH_SIZE):
    # tache1_classes = ['glove_pair_latex', 'glove_pair_nitrile', 'glove_pair_surgery',
    #                   'glove_single_latex', 'glove_single_nitrile', 'glove_single_surgery']
    # tache2_classes = ['shoe_cover_pair', 'shoe_cover_single']
    # tache3_classes = ['urine_bag', 'gauze', 'medical_cap', 'medical_glasses', 'test_tube']

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

            torch.save(model, f"checkpoint/resnet_{source_task}.pth")
            # if taches_classes == tache1_classes:
            #     torch.save(model, f"checkpoint/resnet_tache1.pth")
            # elif taches_classes == tache2_classes:
            #     torch.save(model, f"checkpoint/resnet_tache2.pth")
            # elif taches_classes == tache3_classes:
            #     torch.save(model, f"checkpoint/resnet_tache3.pth")
            print(f"Model saved with Loss: {min_loss:.4f}")
        print(f"Train Loss: {avg_loss:.4f}, Train Accuracy: {avg_accuracy:.2f}%")


def resnet_model_leep_val(taches_classes_cible, source_task, custom_dataset, device, BATCH_SIZE):
    # if source_task == "Tache1":
    #     model = torch.load('checkpoint/resnet_tache1.pth', weights_only=False)
    # elif source_task == "Tache2":
    #     model = torch.load('checkpoint/resnet_tache2.pth', weights_only=False)
    # elif source_task == "Tache3":
    #     model = torch.load('checkpoint/resnet_tache3.pth', weights_only=False)

    model_filename = f"resnet_{source_task}.pth"
    model_path = os.path.join("checkpoint/", model_filename)
    model = torch.load(model_path, weights_only=False)

    num_classes = len(taches_classes_cible)
    #print(model.fc)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    #print(model.fc)
    model.eval()
    tache_cible_loader = custom_dataset.create_subset_loaders(taches_classes_cible, batch_size=BATCH_SIZE)

    criterion = nn.CrossEntropyLoss()

    val_total_loss, val_total_acc, val_batches = 0, 0, 0
    labels_cible = []
    prediction_cible = []
    with torch.no_grad():
        for val_inputs, val_labels in tache_cible_loader:
            val_mapped_labels = val_labels.clone()
            for old_label, new_label in map_table.items():
                val_mapped_labels[val_mapped_labels == old_label] = new_label

            inputs = val_inputs.to(device)
            labels = val_mapped_labels.to(device)
            # print(f"labels: {labels}")
            labels_cible.append(list(labels.cpu().numpy()))

            val_outputs = model(inputs)
            #print(val_outputs.shape)
        #     val_loss = criterion(val_outputs, labels)
            # _ , preds = torch.max(val_outputs, 1)
            # print(f"preds: {preds}")
            prediction_cible.append(list(val_outputs.cpu().numpy()))
        #
            # val_acc = (preds == labels).sum().item() / labels.size(0)
        #     val_total_loss += val_loss.item()
        #     val_total_acc += val_acc
            # val_batches += 1
        #
        # avg_val_loss = val_total_loss / val_batches
        # avg_val_acc = val_total_acc / val_batches * 100
        # print(f"Val Loss: {avg_val_loss:.4f}, Val Accuracy: {avg_val_acc:.2f}%")


    prediction_cible = np.vstack(prediction_cible)
    labels_cible = np.concatenate(labels_cible)
    return prediction_cible, labels_cible

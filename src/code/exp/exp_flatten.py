import torch
import torch.nn as nn
import torch.optim as optim

from model.model_flatten import ConvModel
from plot import plot_metrics


def exp_flatten(device, custom_dataset):
    training_loader, validation_loader = custom_dataset.create_task_loaders(custom_dataset.dataset.classes, batch_size=16)
    model = ConvModel().to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    for epoch in range(30):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for inputs, labels in training_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = loss_fn(outputs, labels)
            loss.backward()

            optimizer.step()
            running_loss += loss.item()

            _, predicted_labels = torch.max(outputs, 1)
            correct_predictions += (predicted_labels == labels).sum().item()
            total_predictions += labels.size(0)

        train_loss = running_loss / len(training_loader)
        train_losses.append(train_loss)
        train_accuracy = correct_predictions / total_predictions * 100
        train_accuracies.append(train_accuracy)
        print(f'Epoch [{epoch + 1}], Training Loss: {running_loss / len(training_loader):.4f}, Training Accuracy: {train_accuracy:.2f}%')

        model.eval()
        val_running_loss = 0.0
        val_correct_predictions = 0
        val_total_predictions = 0

        with torch.no_grad():
            for inputs, labels in validation_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)

                loss = loss_fn(outputs, labels)
                val_running_loss += loss.item()

                _, predicted_labels = torch.max(outputs, 1)
                val_correct_predictions += (predicted_labels == labels).sum().item()
                val_total_predictions += labels.size(0)

        val_loss = val_running_loss / len(validation_loader)
        val_losses.append(val_loss)
        val_accuracy = val_correct_predictions / val_total_predictions * 100
        val_accuracies.append(val_accuracy)
        print(f'Epoch [{epoch + 1}], Validation Loss: {val_running_loss / len(validation_loader):.4f}, Validation Accuracy: {val_accuracy:.2f}%')

    epochs = range(1, len(train_losses) + 1)
    plot_metrics(
        epochs=epochs,
        train_values=train_losses,
        val_values=val_losses,
        ylabel="Loss",
        title="Training and Validation Loss",
        train_label="Training",
        val_label="Validation",
        ylim=(-0.25, 2)
        # save_path="training_validation_loss.png"
    )

    epochs = range(1, len(val_losses) + 1)
    plot_metrics(
        epochs=epochs,
        train_values=train_accuracies,
        val_values=val_accuracies,
        ylabel="Accuracy",
        title="Training and Validation Accuracy",
        train_label="Training",
        val_label="Validation",
        ylim=(30, 105)
        # save_path="training_validation_accuracy.png"
    )

    print("Experiment 1 Finish !")
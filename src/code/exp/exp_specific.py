import torch
import torch.nn as nn
import torch.optim as optim

from model.model_specific import Model
from plot import plot_metrics, plot_confusion_matrix


def exp_specific(device, custom_dataset, tache1_classes, tache2_classes, tache3_classes, BATCH_SIZE):
    tache1_loader_train, tache1_loader_val = custom_dataset.create_task_loaders(tache1_classes, batch_size=BATCH_SIZE)
    tache2_loader_train, tache2_loader_val = custom_dataset.create_task_loaders(tache2_classes, batch_size=BATCH_SIZE)
    tache3_loader_train, tache3_loader_val = custom_dataset.create_task_loaders(tache3_classes, batch_size=BATCH_SIZE)

    num_epochs = 30
    learning_rate = 0.001

    for task_idx, (train_loader, val_loader, task_classes) in enumerate([
        (tache1_loader_train, tache1_loader_val, tache1_classes),
        (tache2_loader_train, tache2_loader_val, tache2_classes),
        (tache3_loader_train, tache3_loader_val, tache3_classes)
    ]):
        model = Model().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        print(f"\n=== Task {task_idx + 1} ===")
        task_train_losses, task_train_accuracies = [], []
        task_val_losses, task_val_accuracies = [], []

        for epoch in range(num_epochs):
            print(f"\n--- Epoch {epoch + 1}/{num_epochs} for Task {task_idx + 1} ---")
            model.train()
            total_loss, total_acc = 0, 0
            total_batches = 0

            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                _, preds = torch.max(outputs, 1)
                acc = (preds == labels).sum().item() / labels.size(0)
                total_loss += loss.item()
                total_acc += acc
                total_batches += 1

            avg_train_loss = total_loss / total_batches
            avg_train_acc = total_acc / total_batches * 100
            task_train_losses.append(avg_train_loss)
            task_train_accuracies.append(avg_train_acc)
            print(f"Train Loss: {avg_train_loss:.4f}, "
                  f"Train Accuracy: {avg_train_acc:.2f}%")

            model.eval()
            val_loss, val_acc = 0, 0
            val_batches = 0
            all_labels = []
            all_predictions = []

            with torch.no_grad():
                offset = sum(len(task) for task in [tache1_classes, tache2_classes][:task_idx])
                print(f"Task {task_idx + 1}: Offset = {offset}")

                for val_inputs, val_labels in val_loader:
                    val_inputs = val_inputs.to(device)
                    val_labels = val_labels.to(device)
                    val_outputs = model(val_inputs)

                    val_loss += criterion(val_outputs, val_labels).item()
                    _, val_preds = torch.max(val_outputs, 1)
                    val_acc += (val_preds == val_labels).sum().item() / val_labels.size(0)
                    val_batches += 1

                    if task_idx == 1:  # Task 2
                        val_labels = val_labels - 6
                        val_preds = val_preds - 6
                    elif task_idx == 2:  # Task 3
                        val_labels = val_labels - 8
                        val_preds = val_preds - 8

                    # print(f"Batch adjusted labels: {val_labels.cpu().numpy()}")
                    # print(f"Batch adjusted predictions: {val_preds.cpu().numpy()}")
                    all_labels.extend(val_labels.cpu().numpy())
                    all_predictions.extend(val_preds.cpu().numpy())

            avg_val_loss = val_loss / val_batches
            avg_val_acc = val_acc / val_batches * 100
            task_val_losses.append(avg_val_loss)
            task_val_accuracies.append(avg_val_acc)
            print(f"Validation Loss: {avg_val_loss:.4f}, "
                  f"Validation Accuracy: {avg_val_acc:.2f}%")

        print(f"\nFinal Results for Task {task_idx + 1}:")
        print(f"Average Train Loss: {sum(task_train_losses) / len(task_train_losses):.4f}")
        print(f"Average Train Accuracy: {sum(task_train_accuracies) / len(task_train_accuracies):.2f}%")
        print(f"Average Validation Loss: {sum(task_val_losses) / len(task_val_losses):.4f}")
        print(f"Average Validation Accuracy: {sum(task_val_accuracies) / len(task_val_accuracies):.2f}%")

        plot_confusion_matrix(all_labels, all_predictions, task_classes)

        epochs = list(range(1, num_epochs + 1))
        plot_metrics(
            epochs=epochs,
            train_values=task_train_losses,
            val_values=task_val_losses,
            ylabel="Loss",
            title=f"Task {task_idx + 1} Loss Over Epochs",
            train_label="Training",
            val_label="Validation",
            ylim=(-0.25, 2)
            # save_path=f"task_{task_idx + 1}_loss.png"
        )
        plot_metrics(
            epochs=epochs,
            train_values=task_train_accuracies,
            val_values=task_val_accuracies,
            ylabel="Accuracy (%)",
            title=f"Task {task_idx + 1} Accuracy Over Epochs",
            train_label="Training",
            val_label="Validation",
            ylim=(30, 105)
            # save_path=f"task_{task_idx + 1}_accuracy.png"
        )

    print("Experiment 2 Finished !")
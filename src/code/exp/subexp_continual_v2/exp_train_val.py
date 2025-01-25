import random
import torch
import torch.nn as nn
import torch.optim as optim

from model.model_continual_v2 import ContinueModel
from plot import plot_metric_continue_train_V2, plot_metric_continue_evalu_V2, plot_confusion_matrix, plot_leep_accuracy

replay_buffer = []
BUFFER_SIZE = 500  # Maximum buffer capacity
BUFFER_SAMPLE_SIZE = 16  # The size of each sample from the buffer

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
    12: 2
}


def exp_train_val(device, leep_scores, custom_dataset, tasks, BATCH_SIZE):
    train_loaders = []
    val_loaders = []
    for key, value in tasks.items():
        loader_train, loader_val = custom_dataset.create_task_loaders(value, batch_size=BATCH_SIZE)
        train_loaders.append(loader_train)
        val_loaders.append(loader_val)

    class_input_dim = 8 * (400 - 4) * (640 - 4)
    learning_rate = 0.001
    num_epochs = 10
    min_val_loss = float('inf')

    train_losses_t1 = []
    train_accuracies_t1 = []
    train_losses_t2 = []
    train_accuracies_t2 = []
    train_losses_t3 = []
    train_accuracies_t3 = []
    train_losses_t4 = []
    train_accuracies_t4 = []

    Accuracy = {}
    # Instantiate the model

    for task_idx, train_loader in enumerate(train_loaders):
        print(f"\n=== Task {task_idx + 1} ===")
        model = ContinueModel(class_input_dim=class_input_dim).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        for epoch in range(num_epochs):
            print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")

            model.train()
            total_loss, total_acc = 0, 0
            total_batches = 0

            for inputs, labels in train_loader:

                mapped_labels = labels.clone()
                for old_label, new_label in map_table.items():
                    mapped_labels[mapped_labels == old_label] = new_label

                inputs = inputs.to(device)
                labels = mapped_labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs, task_idx)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                # print("pred",preds)
                total_acc += (preds == labels).sum().item() / labels.size(0)
                total_batches += 1

            avg_loss = total_loss / total_batches
            avg_accuracy = total_acc / total_batches * 100
            if task_idx == 0:
                train_losses_t4.append(avg_loss)
                train_accuracies_t4.append(avg_accuracy)
            elif task_idx == 1:
                train_losses_t2.append(avg_loss)
                train_accuracies_t2.append(avg_accuracy)
            elif task_idx == 2:
                train_losses_t3.append(avg_loss)
                train_accuracies_t3.append(avg_accuracy)
            else:
                train_losses_t1.append(avg_loss)
                train_accuracies_t1.append(avg_accuracy)
            print(f"Train Loss: {avg_loss:.4f}, Train Accuracy: {avg_accuracy:.2f}%")

        print(f"\n=== Validation after Training Task {task_idx + 1} ===")
        model.eval()

        for eval_task_idx, val_loader in enumerate(val_loaders):
            if task_idx != eval_task_idx:
                val_total_loss, val_total_acc, val_batches = 0, 0, 0
                with torch.no_grad():
                    for val_inputs, val_labels in val_loader:
                        mapped_labels = val_labels.clone()
                        for old_label, new_label in map_table.items():
                            mapped_labels[mapped_labels == old_label] = new_label

                        inputs = val_inputs.to(device)
                        labels = mapped_labels.to(device)
                        # print("labels",labels)
                        val_outputs = model(inputs, eval_task_idx)

                        val_loss = criterion(val_outputs, labels)

                        _, preds = torch.max(val_outputs, 1)
                        # print("preds",preds)

                        val_acc = (preds == labels).sum().item() / labels.size(0)
                        val_total_loss += val_loss.item()
                        val_total_acc += val_acc
                        val_batches += 1

                avg_val_loss = val_total_loss / val_batches
                avg_val_acc = val_total_acc / val_batches * 100
                Accuracy[(task_idx, eval_task_idx)] = avg_val_acc
                print(f"Val Loss: {avg_val_loss:.4f}, Val Accuracy: {avg_val_acc:.2f}%")

    plot_leep_accuracy(Accuracy, leep_scores)
    print("Experiment train and validation Finished !")

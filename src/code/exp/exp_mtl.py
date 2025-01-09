import torch
import torch.nn as nn
import torch.optim as optim

from model.model_mtl import TotalModel
from plot import plot_metrics, plot_confusion_matrix


def exp_mtl(device, custom_dataset, tache1_classes, tache2_classes, tache3_classes, BATCH_SIZE):
    tache1_loader_train, tache1_loader_val = custom_dataset.create_task_loaders(tache1_classes, batch_size=23)
    tache2_loader_train, tache2_loader_val = custom_dataset.create_task_loaders(tache2_classes, batch_size=8)
    tache3_loader_train, tache3_loader_val = custom_dataset.create_task_loaders(tache3_classes, batch_size=20)

    class_input_dim = 8 * (40 - 4) * (64 - 4)
    learning_rate = 0.001
    num_epochs = 30

    # best_model_path = '/Users/jiaqifeng/PycharmProjects/Python_RD/checkpoint/best_model_mtl.pth'
    min_val_loss = float('inf')

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    all_labels = []
    all_predictions = []

    # Instantiate the model
    model = TotalModel(class_input_dim=class_input_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")
        model.train()

        total_loss = 0
        total_acc = 0
        total_batches = 0

        task_loaders = [tache1_loader_train, tache2_loader_train, tache3_loader_train]
        task_iters = [iter(loader) for loader in task_loaders]

        for iteration in range(67):
            loss = 0

            # Traverse the iterator of each task and get a batch
            batch_data = [next(task_iter) for task_iter in task_iters]
            optimizer.zero_grad()

            for task_idx, (inputs, labels) in enumerate(batch_data):
                inputs = inputs.to(device)
                labels = labels.to(device)

                if task_idx == 1:  # Task 2
                    labels = labels - 6
                elif task_idx == 2:  # Task 3
                    labels = labels - 8

                outputs = model(inputs, task_idx)

                loss += criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                acc = (preds == labels).sum().item() / labels.size(0)
                total_acc += acc
                total_batches += 1

            loss.backward()
            optimizer.step()

            total_loss += loss

        avg_train_loss = (total_loss).item() / total_batches
        avg_train_acc = total_acc / total_batches * 100
        train_losses.append(avg_train_loss)
        train_accuracies.append(avg_train_acc)
        print(f"Train Loss: {avg_train_loss:.4f}, "
              f"Train Accuracy: {avg_train_acc:.2f}%")

        model.eval()
        val_loss, val_acc = 0, 0
        val_batches = 0
        all_preds_overall = []
        all_labels_overall = []

        task_loaders_val = [tache1_loader_val, tache2_loader_val, tache3_loader_val]
        task_iters_val = [iter(loader) for loader in task_loaders]

        with torch.no_grad():
            for iteration in range(17):
                batch_data_val = [next(task_iter_val) for task_iter_val in task_iters_val]

                for task_idx, (val_inputs, val_labels) in enumerate(batch_data_val):
                    val_inputs = val_inputs.to(device)
                    val_labels = val_labels.to(device)

                    if task_idx == 1:  # Task 2
                        val_labels = val_labels - 6
                    elif task_idx == 2:  # Task 3
                        val_labels = val_labels - 8

                    val_outputs = model(val_inputs, task_idx)
                    val_loss += criterion(val_outputs, val_labels).item()

                    _, val_preds = torch.max(val_outputs, 1)
                    val_acc += (val_preds == val_labels).sum().item() / val_labels.size(0)
                    val_batches += 1

                    if task_idx == 1:  # Task 2
                        val_labels = val_labels + 6
                        val_preds = val_preds + 6
                    elif task_idx == 2:  # Task 3
                        val_labels = val_labels + 8
                        val_preds = val_preds + 8
                    all_preds_overall.extend(val_preds.cpu().numpy())
                    all_labels_overall.extend(val_labels.cpu().numpy())

        avg_val_loss = val_loss / val_batches
        avg_val_acc = val_acc / val_batches * 100
        val_losses.append(avg_val_loss)
        val_accuracies.append(avg_val_acc)
        print(f"Validation Loss: {avg_val_loss:.4f}, "
              f"Validation Accuracy: {avg_val_acc:.2f}%")

        if avg_val_loss < min_val_loss:
            min_val_loss = avg_val_loss
            # torch.save(model.state_dict(), best_model_path)
            print(f"Model saved with Validation Loss: {min_val_loss:.4f}")

    class_names = custom_dataset.dataset.classes
    plot_confusion_matrix(all_preds_overall, all_labels_overall, class_names)

    epochs = range(1, num_epochs + 1)
    plot_metrics(
        epochs=epochs,
        train_values=train_losses,
        val_values=val_losses,
        ylabel="Loss",
        title="Training and Validation Loss",
        ylim=(-0.25, 2)
    )

    plot_metrics(
        epochs=epochs,
        train_values=train_accuracies,
        val_values=val_accuracies,
        ylabel="Accuracy (%)",
        title="Training and Validation Accuracy",
        ylim=(30, 105)
    )

    print("Experiment 3 Finished !")
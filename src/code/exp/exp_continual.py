import random
import torch
import torch.nn as nn
import torch.optim as optim

from model.model_continual import ContinueModel
from plot import plot_metric_continue_train, plot_metric_continue_evalu, plot_confusion_matrix

replay_buffer = []
BUFFER_SIZE = 500  # Maximum buffer capacity
BUFFER_SAMPLE_SIZE = 64  # The size of each sample from the buffer


# def add_to_replay_buffer(inputs, labels):
    # global replay_buffer
    # inputs, labels = inputs.detach().cpu(), labels.detach().cpu()

    # Iterate over each sample in the Batch and add them to the buffer one by one
    # for i in range(inputs.size(0)):
        # if len(replay_buffer) >= BUFFER_SIZE:
            # replay_buffer.pop(0)  # If the buffer is full, remove the oldest sample
        # replay_buffer.append((inputs[i], labels[i]))


# def sample_from_replay_buffer():
    # global replay_buffer
    # if len(replay_buffer) == 0:
        # return None, None

    # Randomly sample BUFFER_SAMPLE_SIZE samples
    # sampled = random.sample(replay_buffer, min(len(replay_buffer), BUFFER_SAMPLE_SIZE))
    # sampled_inputs, sampled_labels = zip(*sampled)
    # return torch.stack(sampled_inputs), torch.tensor(sampled_labels)


def exp_continual(device, custom_dataset, tache1_classes, tache2_classes, tache3_classes, BATCH_SIZE):
    tache1_loader_train, tache1_loader_val = custom_dataset.create_task_loaders(tache1_classes, batch_size=BATCH_SIZE)
    tache2_loader_train, tache2_loader_val = custom_dataset.create_task_loaders(tache2_classes, batch_size=BATCH_SIZE)
    tache3_loader_train, tache3_loader_val = custom_dataset.create_task_loaders(tache3_classes, batch_size=BATCH_SIZE)

    train_loaders = [tache1_loader_train, tache2_loader_train, tache3_loader_train]
    val_loaders = [tache1_loader_val, tache2_loader_val, tache3_loader_val]

    class_input_dim = 8 * (40 - 4) * (64 - 4)
    learning_rate = 0.001
    num_epochs = 30
    min_val_loss = float('inf')

    train_losses_t1 = []
    train_accuracies_t1 = []
    train_losses_t2 = []
    train_accuracies_t2 = []
    train_losses_t3 = []
    train_accuracies_t3 = []
    val_losses_t1 = []
    val_accuracies_t1 = []
    val_losses_t2 = []
    val_accuracies_t2 = [0]
    val_losses_t3 = []
    val_accuracies_t3 = [0, 0]

    # Instantiate the model
    model = ContinueModel(class_input_dim=class_input_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for task_idx, train_loader in enumerate(train_loaders):
        print(f"\n=== Task {task_idx + 1} ===")

        for epoch in range(num_epochs):
            print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")

            model.train()
            total_loss, total_acc = 0, 0
            total_batches = 0

            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                if task_idx == 1:  # Task 2
                    labels = labels - 6
                elif task_idx == 2:  # Task 3
                    labels = labels - 8

                # Sampling from a buffer
                # replay_inputs, replay_labels = sample_from_replay_buffer()
                # print("replay_inputs:\n",replay_inputs)
                # print("replay_labels:\n",replay_labels)

                # If the buffer is not empty, concatenate the buffer samples with the current task samples
                # if replay_inputs is not None:
                    # replay_inputs = replay_inputs.to(device)
                    # replay_labels = replay_labels.to(device)
                    # inputs = torch.cat([inputs, replay_inputs])
                    # labels = torch.cat([labels, replay_labels])

                optimizer.zero_grad()
                outputs = model(inputs, task_idx)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                total_acc += (preds == labels).sum().item() / labels.size(0)
                total_batches += 1

                # Add buffer
                # add_to_replay_buffer(inputs.detach().cpu(), labels.detach().cpu())

            avg_loss = total_loss / total_batches
            avg_accuracy = total_acc / total_batches * 100
            if task_idx == 0:
                train_losses_t1.append(avg_loss)
                train_accuracies_t1.append(avg_accuracy)
            elif task_idx == 1:
                train_losses_t2.append(avg_loss)
                train_accuracies_t2.append(avg_accuracy)
            else:
                train_losses_t3.append(avg_loss)
                train_accuracies_t3.append(avg_accuracy)
            print(f"Train Loss: {avg_loss:.4f}, Train Accuracy: {avg_accuracy:.2f}%")

        print(f"\n=== Validation after Training Task {task_idx + 1} ===")
        model.eval()
        all_preds_overall = []
        all_labels_overall = []

        for eval_task_idx, val_loader in enumerate(val_loaders[:task_idx + 1]):
            val_total_loss, val_total_acc, val_batches = 0, 0, 0
            with torch.no_grad():
                for val_inputs, val_labels in val_loader:
                    inputs = val_inputs.to(device)
                    labels = val_labels.to(device)

                    if eval_task_idx == 1:  # Task 2
                        labels = labels - 6
                    elif eval_task_idx == 2:  # Task 3
                        labels = labels - 8

                    val_outputs = model(inputs, eval_task_idx)
                    val_loss = criterion(val_outputs, labels)

                    _, preds = torch.max(val_outputs, 1)
                    val_acc = (preds == labels).sum().item() / labels.size(0)
                    val_total_loss += val_loss.item()
                    val_total_acc += val_acc
                    val_batches += 1

                    if eval_task_idx == 1:  # Task 2
                        labels = labels + 6
                        preds = preds + 6
                    elif eval_task_idx == 2:  # Task 3
                        labels = labels + 8
                        preds = preds + 8
                    all_preds_overall.extend(preds.cpu().numpy())
                    all_labels_overall.extend(labels.cpu().numpy())

            avg_val_loss = val_total_loss / val_batches
            avg_val_acc = val_total_acc / val_batches * 100
            if eval_task_idx == 0:
                val_losses_t1.append(avg_val_loss)
                val_accuracies_t1.append(avg_val_acc)
            elif eval_task_idx == 1:
                val_losses_t2.append(avg_val_loss)
                val_accuracies_t2.append(avg_val_acc)
            else:
                val_losses_t3.append(avg_val_loss)
                val_accuracies_t3.append(avg_val_acc)
            print(f"Val Loss: {avg_val_loss:.4f}, Val Accuracy: {avg_val_acc:.2f}%")

            # if avg_val_loss < min_val_loss:
                # min_val_loss = avg_val_loss
                # torch.save(model.state_dict(), f"/Users/jiaqifeng/PycharmProjects/Python_RD/checkpoint/best_model_task{task_idx}.pth")
                # print(f"Model for Task {task_idx + 1} saved.")

    class_names = custom_dataset.dataset.classes
    plot_confusion_matrix(all_preds_overall, all_labels_overall, class_names)

    epochs = range(1, num_epochs + 1)
    plot_metric_continue_train(
        epochs=epochs,
        train_values_t1=train_losses_t1,
        train_values_t2=train_losses_t2,
        train_values_t3=train_losses_t3,
        ylabel="Loss",
        title="Training Loss",
        ylim=(-0.25, 2)
    )

    plot_metric_continue_train(
        epochs=epochs,
        train_values_t1=train_accuracies_t1,
        train_values_t2=train_accuracies_t2,
        train_values_t3=train_accuracies_t3,
        ylabel="Accuracy (%)",
        title="Training  Accuracy",
        ylim=(30, 105)
    )

    num_task = 3
    task = range(1, num_task + 1)
    plot_metric_continue_evalu(
        task=task,
        evalu_values_t1=val_accuracies_t1,
        evalu_values_t2=val_accuracies_t2,
        evalu_values_t3=val_accuracies_t3,
        ylabel="Accuracy (%)",
        title="Validation Accuracy",
        ylim=(30, 105)
    )

    print("Experiment 4 Finished !")
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# matplotlib.use('TkAgg')
matplotlib.use('MacOSX')  # Using this if in MacOS


def plot_metrics(epochs, train_values, val_values, ylabel, title, train_label="Training", val_label="Validation", save_path=None, ylim=None, ymin=None, ymax=None):
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_values, label=f"{train_label} {ylabel}", marker='o')
    plt.plot(epochs, val_values, label=f"{val_label} {ylabel}", marker='o')
    plt.xlabel("Epochs")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid()

    if ylim:
        plt.ylim(ylim)

    if ymin:
        plt.ylim(bottom=ymin)

    plt.show()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")


def plot_metric_continue_train(epochs, train_values_t1, train_values_t2, train_values_t3, ylabel, title, save_path=None, ylim=None, ymin=None, ymax=None):
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_values_t1, label=f"Task1 {ylabel}", marker='o')
    plt.plot(epochs, train_values_t2, label=f"Task2 {ylabel}", marker='o')
    plt.plot(epochs, train_values_t3, label=f"Task3 {ylabel}", marker='o')
    plt.xlabel("Epochs")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid()

    if ylim:
        plt.ylim(ylim)

    if ymin:
        plt.ylim(bottom=ymin)

    plt.show()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")


def plot_metric_continue_evalu(task, evalu_values_t1, evalu_values_t2, evalu_values_t3, ylabel, title, save_path=None, ylim=None, ymin=None, ymax=None):
    plt.figure(figsize=(10, 5))
    evalu_values_t2_modified = evalu_values_t2.copy()
    evalu_values_t2_modified[0] = np.nan
    evalu_values_t3_modified = evalu_values_t3.copy()
    evalu_values_t3_modified[0] = np.nan
    evalu_values_t3_modified[1] = np.nan
    plt.plot(task, evalu_values_t1, label=f"Version1 {ylabel}", marker='o')
    plt.plot(task, evalu_values_t2_modified, label=f"Version2 {ylabel}", marker='o')
    plt.plot(task, evalu_values_t3_modified, label=f"Version3 {ylabel}", marker='o')
    plt.xlabel("Task")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.xticks(np.arange(1, 3, 1))

    if ylim:
        plt.ylim(ylim)

    if ymin:
        plt.ylim(bottom=ymin)

    plt.show()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
import matplotlib
import matplotlib.pyplot as plt

# matplotlib.use('TkAgg')
matplotlib.use('MacOSX')  # Using this if in MacOS

def plot_metrics(epochs, train_values, val_values, ylabel, title, train_label="Training", val_label="Validation", save_path=None):
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_values, label=f"{train_label} {ylabel}", marker='o')
    plt.plot(epochs, val_values, label=f"{val_label} {ylabel}", marker='o')
    plt.xlabel("Epochs")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
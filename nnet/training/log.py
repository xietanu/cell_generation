"""Log of training and validation loss and accuracy."""
import matplotlib.pyplot as plt


class LossAccLog:
    """Log of training and validation loss and accuracy."""

    def __init__(self):
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []

    def most_recent_training(self) -> tuple[str, str]:
        """Return the most recent training loss and accuracy."""
        if not self.train_loss:
            return "-", "-"
        loss = f"{self.train_loss[-1]:.4f}"
        if self.train_acc:
            acc = f"{self.train_acc[-1]:.4%}"
        else:
            acc = "-"
        return loss, acc

    def most_recent_validation(self) -> tuple[str, str]:
        """Return the most recent validation loss and accuracy."""
        if not self.val_loss:
            return "-", "-"
        loss = f"{self.val_loss[-1]:.4f}"
        if self.val_acc:
            acc = f"{self.val_acc[-1]:.1%}"
        else:
            acc = "-"
        return loss, acc

    def add_step(self, *, train_loss, val_loss, train_acc, val_acc):
        """Add a step to the log."""
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)
        if train_acc:
            self.train_acc.append(train_acc)
        if val_acc:
            self.val_acc.append(val_acc)

    def plot(self):
        """Plot the log."""
        fig, _ = self.get_plot_fig()
        plt.show()

    def get_plot_fig(self):
        """Return a figure with the log."""
        if len(self.train_acc + self.val_acc) == 0:
            plt.plot(self.train_loss, label="train")
            plt.plot(self.val_loss, label="val")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid()
            return plt.gcf(), plt.gca()
        fig, ax = plt.subplots(1, 2, figsize=(16, 6))
        ax[0].plot(self.train_loss, label="train")
        ax[0].plot(self.val_loss, label="val")
        ax[0].set_xlabel("Epochs")
        ax[0].set_ylabel("Loss")
        ax[0].legend()
        ax[0].grid()
        ax[1].plot(self.train_acc, label="train")
        ax[1].plot(self.val_acc, label="val")
        ax[1].set_xlabel("Epochs")
        ax[1].set_ylabel("Accuracy")
        ax[1].legend()
        ax[1].grid()
        return fig, ax

"""Log of training and validation loss and accuracy."""
import matplotlib.pyplot as plt


class AdvLossAccLog:
    """Log of training and validation loss and accuracy."""

    def __init__(self):
        self.train_gen_loss = []
        self.train_dis_loss = []

    def most_recent_training(self) -> tuple[str, str]:
        """Return the most recent training loss and accuracy."""
        if not self.train_gen_loss:
            return "-", "-"
        loss = f"{self.train_gen_loss[-1]:.4f}"
        if self.train_dis_loss:
            acc = f"{self.train_dis_loss[-1]:.4f}"
        else:
            acc = "-"
        return loss, acc

    def add_step(self, *, train_gen_loss, train_dis_loss):
        """Add a step to the log."""
        self.train_gen_loss.append(train_gen_loss)
        self.train_dis_loss.append(train_dis_loss)

    def plot(self):
        """Plot the log."""
        fig, _ = self.get_plot_fig()
        fig.show()

    def get_plot_fig(self):
        """Return a figure with the log."""
        fig, ax = plt.subplots(1, 1, figsize=(16, 6))
        ax.plot(self.train_gen_loss, label="Generator")
        ax.plot(self.train_dis_loss, label="Discriminator")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Generator Loss")
        ax.legend()
        ax.grid()
        return fig, ax

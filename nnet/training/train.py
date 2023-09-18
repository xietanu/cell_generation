import os
from typing import Callable, Optional
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import nnet.training


class Trainer:
    """Class to train a neural network."""

    def __init__(
        self,
        model: nn.Module,
        lr=0.001,
        *,
        optimiser: type = optim.Adam,
        criterion: Callable = nn.CrossEntropyLoss(),
        accuracy_measure: Optional[Callable] = None,
    ):
        self.model = model
        self.optimiser = optimiser(model.parameters(), lr=lr)
        self.criterion = criterion
        self.accuracy_measure = accuracy_measure
        self.lr = lr
        self.log = nnet.training.LossAccLog()
        self.epoch = 0

    def train(
        self,
        train_loader,
        val_loader,
        epochs,
        device: str = "cuda",
        *,
        path: Optional[str] = None,
        favour_acc: bool = False,
    ):
        """Train the model for the given number of epochs."""
        if favour_acc and self.accuracy_measure is None:
            raise ValueError("You must provide an accuracy measure to favour accuracy.")

        model = self.model.to(device)

        best_val_loss = np.inf
        best_val_acc = 0

        progress = tqdm(
            range(epochs),
            desc=f"Epoch 0/{epochs} done |",
            leave=True,
        )

        self._update_progress(progress, 0, epochs)

        for epoch in progress:
            train_loss, train_acc = self.train_one_epoch(
                model,
                device,
                train_loader,
            )
            val_loss, val_acc = nnet.training.score(
                model,
                val_loader,
                self.criterion,
                self.accuracy_measure,
                device=device,
            )

            self.log.add_step(
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
            )

            if path:
                if val_loss < best_val_loss and not favour_acc:
                    best_val_loss = val_loss
                    self.save_state(path, val_loss)
                elif favour_acc and val_acc and val_acc > best_val_acc:
                    best_val_acc = val_acc
                    self.save_state(path, val_loss)

            self.epoch += 1

            self._update_progress(progress, epoch + 1, epochs)

    def _update_progress(
        self,
        progress,
        epoch,
        n_epochs,
    ):
        """Update the progress bar."""
        train_loss, train_acc = self.log.most_recent_training()
        val_loss, val_acc = self.log.most_recent_validation()

        progress.set_description(
            f"Epoch {epoch}/{n_epochs} done |"
            f" train loss: {train_loss}"
            f" train acc: {train_acc} |"
            f" val loss: {val_loss}"
            f" val acc : {val_acc}."
        )

    def train_one_epoch(self, model, device, train_loader):
        """Train the model for one epoch."""
        model.train()
        criterion = self.criterion

        loss_step = []
        total, train_acc = 0, 0.0

        for inp_data, labels in train_loader:
            if isinstance(inp_data, list):
                inp_data = [data.to(device) for data in inp_data]
            else:
                inp_data = inp_data.to(device)
            if isinstance(labels, list):
                labels = [data.to(device) for data in labels]
            else:
                labels = labels.to(device)

            outputs = model(inp_data)
            loss = criterion(outputs, labels)
            # loss.requires_grad = True
            loss.backward()

            self.optimiser.step()
            self.optimiser.zero_grad()

            if self.accuracy_measure is not None:
                with torch.no_grad():
                    batch_acc = self.accuracy_measure(outputs, labels)
                    train_acc = (batch_acc * len(labels) + train_acc * total) / (
                        total + len(labels)
                    )
                    total += len(labels)

            loss_step.append(loss.item())

        loss_curr_epoch = np.mean(loss_step)

        return loss_curr_epoch, train_acc if self.accuracy_measure else None

    def save_state(self, path, val_loss):
        """Save the state of the model."""
        try:
            os.remove(path)
        except OSError:
            pass
        torch.save(
            {
                "epoch": self.epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimiser.state_dict(),
                "loss": val_loss,
            },
            path,
        )

    def load_state(self, path):
        """Load the state of the model."""
        checkpoint = torch.load(path)
        self.model.to("cuda")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimiser.load_state_dict(checkpoint["optimizer_state_dict"])
        print(
            f"Model {path} is loaded from epoch {checkpoint['epoch']} , loss {checkpoint['loss']:.4f}"
        )

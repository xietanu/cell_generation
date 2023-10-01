import os
from typing import Callable, Optional
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import nnet.training

REAL_LABEL = 1
FAKE_LABEL = 0


class GANTrainer:
    """Class to train a neural network."""

    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        means: np.ndarray,
        stds: np.ndarray,
        lr_gen=0.001,
        lr_disc=0.001,
        *,
        optimiser: type = optim.Adam,
        criterion: Callable = nn.BCELoss(),
    ):
        self.generator = generator
        self.gen_optimiser = optimiser(generator.parameters(), lr=lr_gen)
        self.discriminator = discriminator
        self.disc_optimiser = optimiser(discriminator.parameters(), lr=lr_disc)
        self.criterion = criterion
        self.lr_gen = lr_gen
        self.lr_disc = lr_disc
        self.log = nnet.training.AdvLossAccLog()
        self.epoch = 0
        self.means = means
        self.stds = stds

    def train(
        self,
        train_loader,
        val_loader,
        epochs,
        device: torch.device = torch.device("cuda:0"),
        *,
        path: Optional[str] = None,
        noise: float = 0.0,
        gen_train_thresh: float | None = 2.5,
        dis_train_thresh: float | None = 2.5,
        bookmark_every: int = 10,
    ):
        """Train the model for the given number of epochs."""
        generator = self.generator.to(device)
        discriminator = self.discriminator.to(device)

        progress = tqdm(
            range(epochs),
            desc=f"Epoch 0/{epochs} done |",
            leave=True,
        )

        start_epoch = self.epoch

        self._update_progress(progress, 0, epochs)

        train_gen_loss, train_dis_loss = self.score_one_epoch(
            generator,
            discriminator,
            device,
            val_loader,
        )

        trained_gen, trained_dis = 0, 0

        for _ in progress:
            tg, td, tgl, tdl = self.train_one_epoch(
                generator,
                discriminator,
                device,
                train_loader,
                noise,
                train_gen_loss,
                train_dis_loss,
                gen_train_thresh,
                dis_train_thresh,
            )

            if tgl != 0:
                train_gen_loss = tgl
            if tdl != 0:
                train_dis_loss = tdl

            trained_gen += tg
            trained_dis += td

            val_gen_loss, val_dis_loss = self.score_one_epoch(
                generator, discriminator, device, val_loader, noise
            )

            self.log.add_step(
                train_gen_loss=val_gen_loss,
                train_dis_loss=val_dis_loss,
            )

            self.epoch += 1

            if path and self.epoch % bookmark_every == 0:
                self.save_state(path + f"_{self.epoch}.pt", val_gen_loss)

            self._update_progress(
                progress,
                self.epoch,
                start_epoch + epochs,
                trained_gen,
                trained_dis,
                train_gen_loss,  # type: ignore
                train_dis_loss,  # type: ignore
            )

            if train_dis_loss < 0:
                return

    def _update_progress(
        self,
        progress,
        epoch,
        n_epochs,
        trained_gen=0,
        trained_dis=0,
        train_gen_loss=0.0,
        train_dis_loss=0.0,
    ):
        """Update the progress bar."""
        val_gen_loss, val_dis_loss = self.log.most_recent_training()

        progress.set_description(
            f"Epoch {epoch}/{n_epochs} |"
            f" val loss (gen/dis): {val_gen_loss}/{val_dis_loss} |"
            f" train loss: {train_gen_loss:.2f}({trained_gen})/{train_dis_loss:.2f}({trained_dis})"
        )

    def train_one_epoch(
        self,
        generator,
        discriminator,
        device,
        train_loader,
        label_noise,
        gen_loss_last,
        dis_loss_last,
        gen_train_thresh: float | None = 2.5,
        dis_train_thresh: float | None = 2.5,
    ):
        """Train the model for one epoch."""
        generator.train()
        discriminator.train()
        criterion = self.criterion

        trained_gen, trained_dis = 0, 0

        dis_loss_step = []
        gen_loss_step = []

        for inp_data in train_loader:
            real_cpu = inp_data[0]
            if isinstance(real_cpu, (tuple, list)):
                real_cpu = real_cpu[0]

            real_cpu = real_cpu.to(device)

            batch_size = real_cpu.size(0)

            noise = torch.Tensor(
                np.random.normal(self.means, self.stds, (batch_size, len(self.means))),
            ).to(device)

            fake = generator(noise)

            if isinstance(fake, tuple):
                fake = fake[0]

            if (
                dis_train_thresh is None
                or gen_loss_last < dis_train_thresh
                or (
                    gen_loss_last >= dis_train_thresh
                    and dis_loss_last >= dis_train_thresh
                )
            ):
                trained_dis = 1
                # gaus_noise = torch.Tensor(
                #     np.random.normal(0, 0.5, real_cpu.shape),
                # ).to(device) / (self.epoch/50 + 1)

                # perc_noise = 1 / (self.epoch / 50 + 1)
                # perc_real = 1 - perc_noise

                # # real_cpu = real_cpu * perc_real + gaus_noise * perc_noise
                # real_cpu = gaus_noise

                # if gen_loss < 2.5:
                ###REAL DATA###

                discriminator.zero_grad()

                labels = (
                    torch.randn(batch_size, device=device) * -label_noise + REAL_LABEL
                )

                output = discriminator(real_cpu).view(-1)
                dis_loss_real = criterion(output, labels)

                ###FAKE DATA###

                labels = (
                    torch.randn(batch_size, device=device) * label_noise + FAKE_LABEL
                )

                output = discriminator(fake.detach()).view(-1)

                diss_loss_fake = criterion(output, labels)

                dis_loss = dis_loss_real + diss_loss_fake

                dis_loss.backward()

                dis_loss_step.append(dis_loss.item())

                self.disc_optimiser.step()

            ###GENERATOR###
            if gen_train_thresh is None or dis_loss_last < gen_train_thresh:
                trained_gen = 1
                generator.zero_grad()
                labels = (
                    torch.randn(batch_size, device=device) * -label_noise + REAL_LABEL
                )

                output = discriminator(fake).view(-1)

                gen_loss = criterion(output, labels)
                gen_loss.backward()

                gen_loss_step.append(gen_loss.item())

                self.gen_optimiser.step()

        gen_loss = 0
        dis_loss = 0

        if trained_gen == 1:
            gen_loss = np.mean(gen_loss_step)
        if trained_dis == 1:
            dis_loss = np.mean(dis_loss_step)

        return trained_gen, trained_dis, gen_loss, dis_loss

    def score_one_epoch(
        self,
        generator,
        discriminator,
        device,
        val_loader,
        label_noise=0.0,
    ):
        """Train the model for one epoch."""
        generator.eval()
        discriminator.eval()
        criterion = self.criterion

        gen_loss_step = []
        dis_loss_step = []

        for inp_data in val_loader:
            real_cpu = inp_data[0]

            if isinstance(real_cpu, (tuple, list)):
                real_cpu = real_cpu[0]

            real_cpu = real_cpu.to(device)

            batch_size = real_cpu.size(0)

            ###REAL DATA###

            discriminator.zero_grad()

            labels = (
                torch.ones(batch_size, device=device) * -label_noise / 2 + REAL_LABEL
            )

            output = discriminator(real_cpu).view(-1)
            dis_loss_real = criterion(output, labels)

            ###FAKE DATA###

            noise = torch.Tensor(
                np.random.normal(self.means, self.stds, (batch_size, len(self.means))),
            ).to(device)

            fake = generator(noise)

            if isinstance(fake, tuple):
                fake = fake[0]

            labels = (
                torch.ones(batch_size, device=device) * label_noise / 2 + FAKE_LABEL
            )

            output = discriminator(fake.detach()).view(-1)

            diss_loss_fake = criterion(output, labels)

            dis_loss = dis_loss_real + diss_loss_fake

            dis_loss_step.append(dis_loss.item())

            ###GENERATOR###
            generator.zero_grad()
            labels = (
                torch.ones(batch_size, device=device) * -label_noise / 2 + REAL_LABEL
            )

            output = discriminator(fake).view(-1)

            gen_loss = criterion(output, labels)

            gen_loss_step.append(gen_loss.item())

        gen_loss = np.mean(gen_loss_step)
        dis_loss = np.mean(dis_loss_step)

        return gen_loss, dis_loss

    def save_state(self, path, val_loss):
        """Save the state of the model."""
        try:
            os.remove(path)
        except OSError:
            pass
        torch.save(
            {
                "epoch": self.epoch,
                "gen_state_dict": self.generator.state_dict(),
                "dis_state_dict": self.discriminator.state_dict(),
                "gen_optimizer_state_dict": self.gen_optimiser.state_dict(),
                "dis_optimizer_state_dict": self.disc_optimiser.state_dict(),
                "loss": val_loss,
            },
            path,
        )

    def load_state(self, path):
        """Load the state of the model."""
        checkpoint = torch.load(path)
        self.generator.cuda()
        self.discriminator.cuda()
        self.generator.load_state_dict(checkpoint["gen_state_dict"])
        self.discriminator.load_state_dict(checkpoint["dis_state_dict"])
        self.gen_optimiser.load_state_dict(checkpoint["gen_optimizer_state_dict"])
        self.disc_optimiser.load_state_dict(checkpoint["dis_optimizer_state_dict"])
        self.epoch = checkpoint["epoch"]
        print(
            f"Model {path} is loaded from epoch {checkpoint['epoch']} , loss {checkpoint['loss']:.4f}"
        )

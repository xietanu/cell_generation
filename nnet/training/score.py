"""Scoring function for the training loop."""
from typing import Callable, Optional
import torch


def score(
    model: torch.nn.Module,
    val_loader,
    loss_measure: Callable,
    acc_measure: Optional[Callable],
    device="cuda",
) -> tuple[float, Optional[float]]:
    """Score the model on the validation set."""
    model.eval()

    total, val_acc = 0, 0.0
    loss_step = []

    with torch.no_grad():
        for data, labels in val_loader:
            if isinstance(data, list):
                data = [i.to(device) for i in data]
            else:
                data = data.to(device)
            if isinstance(labels, list):
                labels = [l.to(device) for l in labels]
            else:
                labels = labels.to(device)
            outputs = model(data)

            loss = loss_measure(outputs, labels)

            if acc_measure is not None:
                batch_acc = acc_measure(outputs, labels)
                val_acc = (batch_acc * len(labels) + val_acc * total) / (
                    total + len(labels)
                )
                total += len(labels)

            loss_step.append(loss.item())

    val_loss = torch.tensor(loss_step).mean().cpu().numpy()  # pylint: disable=no-member

    return val_loss, val_acc if acc_measure is not None else None

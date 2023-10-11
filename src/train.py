import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader

from data import Mode
from data_iteration import iterate
from load_config import TrainingCfg
from model import BaseModel


def train(
    cfg: TrainingCfg,
    model: BaseModel,
    dataloaders: dict[str, DataLoader],
    random_rotation: bool,
    single_protein: bool,
    mode: Mode,
    model_path: str,
    device: torch.device,
):
    writer = SummaryWriter("runs/{}".format("TODO"))
    # Baseline optimizer:
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, amsgrad=True)
    best_loss = 1e10  # We save the "best model so far"

    starting_epoch = 0
    if cfg.restart_training != "":
        checkpoint = torch.load("models/" + cfg.restart_training)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        starting_epoch = checkpoint["epoch"]
        best_loss = checkpoint["best_loss"]

    # Training loop (~100 times) over the dataset:
    for epoch in range(starting_epoch, cfg.n_epochs):
        # Train first, Test second:
        for split, dataloader in dataloaders.items():
            # Perform one pass through the data:
            losses, roc_aucs, input_r_values, conv_r_values = iterate(
                model,
                dataloader,
                device,
                optimizer,
                summary_writer=writer,
                epoch_number=epoch,
            )

            # Write down the results using a TensorBoard writer:

            writer.add_scalar(f"loss/{split}", np.mean(losses), epoch)
            writer.add_scalar(f"ROC_AUC/{split}", np.mean(roc_aucs), epoch)

            val = np.array(input_r_values)
            writer.add_scalar(f"Input_R_Values/{split}", np.mean(val[val > 0]), epoch)
            val = np.array(conv_r_values)
            writer.add_scalar(f"Conv_R_Values/{split}", np.mean(val[val > 0]), epoch)

            if split == "Validation":
                # Store validation loss for saving the model
                val_loss = np.mean(losses)

                if val_loss < best_loss:
                    print("Validation loss {}, saving model".format(val_loss))
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "best_loss": best_loss,
                        },
                        model_path + "_epoch{}".format(i),
                    )

                    best_loss = val_loss

    return model

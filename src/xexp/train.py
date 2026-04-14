"""Module with utilities for training and validation"""

from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import (
    BATCH_SIZE,
    RANDOM_SEED,
    collate_fn,
    gene_labels,
    tissue_labels,
    train_dataset,
    valid_dataset,
)
from nn import SetTransformer


class GELoss(nn.Module):
    """Combine standard losses with uncertainty as criterion"""

    def __init__(
        self,
        mse_weight: float = 1.0,
        mae_weight: float = 0.1,
        uncertainty_weight: float = 0.5,
    ):
        super().__init__()
        self.mse_weight = mse_weight
        self.mae_weight = mae_weight
        self.uncertainty_weight = uncertainty_weight

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            predictions: Output from model forward pass
            targets: Ground truth expression
            mask: Optional mask for valid targets
        """
        pred_expressions = predictions["expressions"]
        pred_uncertainties = predictions["uncertainties"]

        if mask is not None:
            pred_expressions = pred_expressions[mask]
            targets = targets[mask]
            pred_uncertainties = pred_uncertainties[mask]

        # Negative Log-Likelihood with Gaussian assumption
        nll = 0.5 * torch.log(pred_uncertainties + 1e-6) + 0.5 * (
            targets - pred_expressions
        ) ** 2 / (pred_uncertainties + 1e-6)
        nll_loss = nll.mean()

        # MSE for point estimate
        mse_loss = F.mse_loss(pred_expressions, targets)

        # MAE for robustness
        mae_loss = F.l1_loss(pred_expressions, targets)

        # Combined loss
        total_loss = (
            self.mse_weight * mse_loss
            + self.uncertainty_weight * nll_loss
            + self.mae_weight * mae_loss
        )

        return {"total": total_loss, "mse": mse_loss, "nll": nll_loss, "mae": mae_loss}


class Trainer:
    """Training loop with validation"""

    def __init__(
        self,
        model: SetTransformer,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.criterion = GELoss()

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        total_loss = 0
        total_mse = 0

        for batch in self.train_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Forward pass
            outputs = self.model(
                obs_tissues=batch["obs_tissues"],
                obs_genes=batch["obs_genes"],
                obs_expressions=batch["obs_expressions"],
                query_tissues=batch["query_tissues"],
                query_genes=batch["query_genes"],
                obs_mask=batch["obs_mask"],
            )

            # Compute loss
            losses = self.criterion(
                outputs,
                batch["targets"],
            )

            # Backward
            self.optimizer.zero_grad()
            losses["total"].backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += losses["total"].item()
            total_mse += losses["mse"].item()

        n = len(self.train_loader)

        return {"loss": total_loss / n, "mse": total_mse / n}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0
        total_mse = 0

        for batch in self.val_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            outputs = self.model(
                obs_tissues=batch["obs_tissues"],
                obs_genes=batch["obs_genes"],
                obs_expressions=batch["obs_expressions"],
                query_tissues=batch["query_tissues"],
                query_genes=batch["query_genes"],
                obs_mask=batch["obs_mask"],
            )

            losses = self.criterion(outputs, batch["targets"])
            total_loss += losses["total"].item()
            total_mse += losses["mse"].item()

        n = len(self.val_loader)

        metrics = {"val_loss": total_loss / n, "val_mse": total_mse / n}

        self.scheduler.step(metrics["val_loss"])

        return metrics


if __name__ == "__main__":
    if not Path("../../results/xexp_weights.pt").exists():
        torch.manual_seed(RANDOM_SEED)

        train_dataloader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
        )

        valid_dataloader = DataLoader(
            valid_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
        )

        model = SetTransformer(
            n_tissues=len(np.unique(tissue_labels)),
            n_genes=len(np.unique(gene_labels)),
            dims=128,
            n_heads=4,
            n_encoder_layers=2,
            n_decoder_layers=2,
        )

        trainer = Trainer(model, train_dataloader)  # , valid_dataloader)

        print("Training...")
        for epoch in range(10):
            metrics = trainer.train_epoch()
            print(
                f"Epoch {epoch+1}: Total loss={metrics['loss']:.4f}, MSE={metrics['mse']:.4f}"
            )

        # print("Validation...")
        # metrics = trainer.validate()
        # print(f"Total loss={metrics['val_loss']:.4f}, MSE={metrics['val_mse']:.4f}")

        # Save weights of trained model
        torch.save(model.state_dict(), "../../results/xexp_weights.pt")

    else:
        print("Model was already trained!")

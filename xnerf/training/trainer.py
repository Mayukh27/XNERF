from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from xnerf.training.losses import classification_losses, total_loss
from xnerf.utils.base import Trainer, move_to_device


class XNerfTrainer(Trainer):
    """Production training loop.

    Inputs:
        model, train/val DatasetLoader, optimizer config.
    Outputs:
        checkpoint files and metrics dict.
    Tensor dimensions:
        consumes batch tensors documented by MalwareManifestDataset.
    Usage:
        trainer = XNerfTrainer(model, train_ds, val_ds)
        trainer.fit()
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset,
        val_dataset=None,
        batch_size: int = 8,
        lr: float = 3e-4,
        epochs: int = 10,
        grad_accum: int = 1,
        num_workers: int = 2,
        checkpoint_dir: str | Path = "checkpoints",
        patience: int = 3,
        device: str | None = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers) if val_dataset else None
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.device.type == "cuda")
        self.epochs = epochs
        self.grad_accum = grad_accum
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.patience = patience

    def _step(self, batch: dict[str, Any], train: bool) -> float:
        batch = move_to_device(batch, self.device)
        with torch.set_grad_enabled(train), torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
            outputs = self.model(batch)
            loss = total_loss(classification_losses(outputs, batch))
        if train:
            self.scaler.scale(loss / self.grad_accum).backward()
        return float(loss.detach().cpu())

    def fit(self) -> dict[str, Any]:
        best = float("inf")
        bad_epochs = 0
        history = []
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            train_loss = 0.0
            for i, batch in enumerate(tqdm(self.train_loader, desc=f"epoch {epoch} train")):
                train_loss += self._step(batch, train=True)
                if (i + 1) % self.grad_accum == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
            val_loss = self.validate() if self.val_loader else train_loss / max(1, len(self.train_loader))
            history.append({"epoch": epoch, "train_loss": train_loss / max(1, len(self.train_loader)), "val_loss": val_loss})
            if val_loss < best:
                best = val_loss
                bad_epochs = 0
                torch.save({"model": self.model.state_dict(), "epoch": epoch, "val_loss": val_loss}, self.checkpoint_dir / "best.pt")
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    break
        return {"best_val_loss": best, "history": history}

    @torch.no_grad()
    def validate(self) -> float:
        self.model.eval()
        total = 0.0
        for batch in tqdm(self.val_loader, desc="validate"):
            total += self._step(batch, train=False)
        return total / max(1, len(self.val_loader))


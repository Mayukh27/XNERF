from __future__ import annotations

import gc
from pathlib import Path
from typing import Any
from uuid import uuid4

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
        resume_from: str | Path | None = None,
        grad_clip: float = 1.0,
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
        self.grad_clip = grad_clip
        self.start_epoch = 1
        self.best_val_loss = float("inf")
        self.bad_epochs = 0
        if resume_from:
            self._load_resume_checkpoint(Path(resume_from))

    def _load_resume_checkpoint(self, checkpoint_path: Path) -> None:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])
        epoch = int(checkpoint.get("epoch", 0))
        self.start_epoch = epoch + 1
        self.best_val_loss = float(checkpoint.get("best_val_loss", checkpoint.get("val_loss", float("inf"))))
        self.bad_epochs = int(checkpoint.get("bad_epochs", 0))
        del checkpoint
        gc.collect()
        print(f"Resuming training from {checkpoint_path} at epoch {self.start_epoch}")

    def _save_checkpoint(self, path: Path, epoch: int, val_loss: float, best_val_loss: float, bad_epochs: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scaler": self.scaler.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "best_val_loss": best_val_loss,
                "bad_epochs": bad_epochs,
            },
            tmp_path,
        )
        try:
            tmp_path.replace(path)
        except PermissionError:
            fallback_path = path.with_name(f"{path.stem}_epoch_{epoch}_{uuid4().hex[:8]}{path.suffix}")
            tmp_path.replace(fallback_path)
            print(
                f"warning: could not replace locked checkpoint {path}; saved {fallback_path} instead",
                flush=True,
            )

    def _diagnostic_value(self, value: Any, limit: int = 8) -> Any:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().flatten()[:limit].tolist()
        if isinstance(value, (list, tuple)):
            return list(value[:limit])
        return value

    def _batch_diagnostics(self, batch: dict[str, Any], batch_idx: int | None) -> dict[str, Any]:
        keys = ("dataset", "label", "arch_id", "family_label", "family", "sha256", "sample_id", "row_index", "path")
        diagnostics: dict[str, Any] = {"batch_idx": batch_idx}
        for key in keys:
            if key in batch:
                diagnostics[key] = self._diagnostic_value(batch[key])
        return diagnostics

    def _raise_nonfinite(self, message: str, batch: dict[str, Any], batch_idx: int | None) -> None:
        raise RuntimeError(f"{message}; batch={self._batch_diagnostics(batch, batch_idx)}")

    def _step(self, batch: dict[str, Any], train: bool, batch_idx: int | None = None) -> float:
        batch = move_to_device(batch, self.device)
        with torch.set_grad_enabled(train), torch.cuda.amp.autocast(enabled=False):
            outputs = self.model(batch)
            for name, value in outputs.items():
                if torch.is_tensor(value) and not torch.isfinite(value).all():
                    self._raise_nonfinite(f"non-finite output detected: {name}", batch, batch_idx)
            losses = classification_losses(outputs, batch)
            for name, value in losses.items():
                if torch.is_tensor(value) and not torch.isfinite(value).all():
                    self._raise_nonfinite(f"non-finite loss term detected: {name}", batch, batch_idx)
            loss = total_loss(losses)
        if not torch.isfinite(loss):
            details = {
                name: float(value.detach().cpu())
                for name, value in losses.items()
                if value.numel() == 1
            }
            self._raise_nonfinite(
                f"non-finite loss detected: total={float(loss.detach().cpu())}, terms={details}",
                batch,
                batch_idx,
            )
        if train:
            self.scaler.scale(loss / self.grad_accum).backward()
        return float(loss.detach().cpu())

    def fit(self) -> dict[str, Any]:
        best = self.best_val_loss
        bad_epochs = self.bad_epochs
        history = []
        if self.start_epoch > self.epochs:
            return {"best_val_loss": best, "history": history, "resumed_from_epoch": self.start_epoch - 1}
        for epoch in range(self.start_epoch, self.epochs + 1):
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            train_loss = 0.0
            for i, batch in enumerate(tqdm(self.train_loader, desc=f"epoch {epoch} train")):
                train_loss += self._step(batch, train=True, batch_idx=i)
                if (i + 1) % self.grad_accum == 0:
                    if self.grad_clip and self.grad_clip > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
            if len(self.train_loader) % self.grad_accum != 0:
                if self.grad_clip and self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
            val_loss = self.validate() if self.val_loader else train_loss / max(1, len(self.train_loader))
            history.append({"epoch": epoch, "train_loss": train_loss / max(1, len(self.train_loader)), "val_loss": val_loss})
            if val_loss < best:
                best = val_loss
                bad_epochs = 0
                self._save_checkpoint(self.checkpoint_dir / "best.pt", epoch, val_loss, best, bad_epochs)
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    self._save_checkpoint(self.checkpoint_dir / "last.pt", epoch, val_loss, best, bad_epochs)
                    print(
                        f"epoch {epoch}: "
                        f"train_loss={history[-1]['train_loss']:.4f} "
                        f"val_loss={val_loss:.4f} "
                        f"best_val_loss={best:.4f}",
                        flush=True,
                    )
                    break
            print(
                f"epoch {epoch}: "
                f"train_loss={history[-1]['train_loss']:.4f} "
                f"val_loss={val_loss:.4f} "
                f"best_val_loss={best:.4f}",
                flush=True,
            )
            self._save_checkpoint(self.checkpoint_dir / "last.pt", epoch, val_loss, best, bad_epochs)
        return {"best_val_loss": best, "history": history}

    @torch.no_grad()
    def validate(self) -> float:
        self.model.eval()
        total = 0.0
        for i, batch in enumerate(tqdm(self.val_loader, desc="validate")):
            total += self._step(batch, train=False, batch_idx=i)
        return total / max(1, len(self.val_loader))


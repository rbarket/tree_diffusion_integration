from __future__ import annotations

import lightning.pytorch as pl
from torch.utils.data import DataLoader

from src.tree_diffusion.tokenizer import TreeDiffusionTokenizer


class TreeDiffusionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        *,
        train_loader: DataLoader,
        val_loader: DataLoader,
        tokenizer: TreeDiffusionTokenizer,
        validation_held_out: bool,
    ) -> None:
        super().__init__()
        self._train_loader = train_loader
        self._val_loader = val_loader
        self.tokenizer = tokenizer
        self.validation_held_out = bool(validation_held_out)

    def train_dataloader(self) -> DataLoader:
        return self._train_loader

    def val_dataloader(self) -> DataLoader:
        return self._val_loader


__all__ = ["TreeDiffusionDataModule"]

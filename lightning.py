import pytorch_lightning as pl
from pytorch_lightning import callbacks
from torch.optim import optimizer
from pathlib import Path
import pandas as pd
from data import *
import torch.nn.functional as F
import torch
from argparse import ArgumentParser
from pytorch_lightning.loggers import WandbLogger
from torch import optim
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import timm

loss_fns = {
    "binary_cross_entropy_with_logits": F.binary_cross_entropy_with_logits,
}


class DogsandCatsModel(pl.LightningModule):
    def __init__(
        self,
        model_name: str = 'resnet50',
        num_classes: int = None,
        loss_fn: str = "binary_cross_entropy",
        lr=1e-4,
        wd=1e-6,
    ):
        super().__init__()

        self.model = timm.create_model(
            model_name=model_name, pretrained=True, num_classes=num_classes)
        self.loss_fn = loss_fns[loss_fn]
        self.lr = lr
        self.accuracy = pl.metrics.Accuracy()
        self.wd = wd

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).view(-1)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).view(-1)
        loss = self.loss_fn(y_hat, y)
        self.log("valid_loss", loss, prog_bar=True)
        self.log("val_acc", self.accuracy(y_hat, y), prog_bar=True)

    def configure_optimizers(self):

        optimizer = optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.wd
        )

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.trainer.max_epochs, 0
        )

        return [optimizer], [scheduler]


@hydra.main(config_path="conf", config_name="config")
def cli_hydra(cfg: DictConfig):
    pl.seed_everything(1234)

    # ------------
    # Log Metrics using Wandb
    # ------------

    wandb_logger = instantiate(cfg.wandb)
    wandb_logger.log_hyperparams(cfg)

    # ------------
    # Create Data Module
    # ------------

    data_module = instantiate(cfg.data)
    data_module.prepare_data()
    data_module.setup()

    # ------------
    # Create Model
    # ------------

    model = instantiate(cfg.model)

    # ------------
    # Training
    # ------------

    trainer = pl.Trainer(
        logger=[wandb_logger], **cfg.trainer
    )

    trainer.fit(model=model, datamodule=data_module)


if __name__ == "__main__":
    cli_hydra()

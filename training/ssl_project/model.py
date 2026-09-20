import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from augment import SSLKorniaAugment
from loss import BarlowTwinsLoss


class ProjectionHead(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=1024, output_dim=2048):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),

            nn.Linear(hidden_dim, output_dim, bias=False),
            nn.BatchNorm1d(output_dim, affine=False),
        )

    def forward(self, x):
        return self.net(x)


class BarlowTwins(pl.LightningModule):
    def __init__(self, encoder, config, num_training_samples):
        super().__init__()

        self.encoder = encoder
        self.config = config

        self.ssl_aug = SSLKorniaAugment(
            image_size=config["data"]["image_size"]
        )

        # Tu suy ra so kenh output cua backbone thay vi hardcode (kenh nay
        # thay doi theo kien truc/weight truyen vao --weight, hardcode da
        # gay loi/nham lan khi doi kien truc nhieu lan truoc do).
        with torch.no_grad():
            encoder_device = next(self.encoder.parameters()).device
            dummy = torch.zeros(
                2, 3, config["data"]["image_size"], config["data"]["image_size"],
                device=encoder_device,
            )
            feat_dim = self.encoder(dummy).shape[1]

        self.projection_head = ProjectionHead(
            input_dim=feat_dim,
            hidden_dim=1024,
            output_dim=config["ssl"]["z_dim"],
        )

        self.loss_fn = BarlowTwinsLoss(
            lambda_coeff=config["ssl"]["lambda_coeff"]
        )

        self.train_iters_per_epoch = math.ceil(
            num_training_samples / config["train"]["batch_size"]
        )

    def shared_step(self, batch):
        x, _ = batch

        x1 = self.ssl_aug(x)
        x2 = self.ssl_aug(x)

        h1 = self.encoder(x1)
        h2 = self.encoder(x2)

        z1 = self.projection_head(h1)
        z2 = self.projection_head(h2)

        loss = self.loss_fn(z1, z2)

        pos_sim = F.cosine_similarity(z1, z2, dim=1).mean()
        emb_var = torch.cat([z1, z2], dim=0).var(dim=0).mean()

        return loss, pos_sim, emb_var

    def training_step(self, batch, batch_idx):
        loss, pos_sim, emb_var = self.shared_step(batch)

        self.log(
            "train_loss",
            loss,
            #on_step=False,
            #on_epoch=True,
            prog_bar=True,
        )

        self.log(
            "train_pos_sim",
            pos_sim,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        self.log(
            "train_emb_var",
            emb_var,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        loss, pos_sim, emb_var = self.shared_step(batch)

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        self.log(
            "val_pos_sim",
            pos_sim,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        self.log(
            "val_emb_var",
            emb_var,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config["train"]["learning_rate"],
            weight_decay=self.config["train"]["weight_decay"],
        )

        return optimizer

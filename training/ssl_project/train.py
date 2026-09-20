from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from config import CONFIG
from datasets import build_datasets
from encoder import build_encoder
from model import BarlowTwins

import argparse
import copy
import os


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--weight",
        type=str,
        default="last.pt",
        help="YOLO weight or yaml file (chi backbone duoc dung), e.g. last.pt, yolo26n-cls.yaml"
    )

    parser.add_argument(
        "--lambda-coeff",
        type=float,
        default=None,
        help="SSL lambda coefficient"
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        # default="/home/agi/thesis_code/ssl_project/checkpoints/ssl_last_lambda0.005/last-v4.ckpt",
        help="Resume checkpoint path (khong truyen = train tu dau tu --weight)"
    )

    return parser.parse_args()


def build_exp_name(weight, lambda_coeff):
    weight_name = os.path.basename(weight)
    weight_name = weight_name.replace(".pt", "").replace(".yaml", "")

    exp_name = f"ssl_{weight_name}_lambda{lambda_coeff}"

    return exp_name


args = parse_args()

# Copy config để tránh sửa trực tiếp CONFIG gốc
config = copy.deepcopy(CONFIG)

# Ghi đè lambda nếu truyền từ command line
if args.lambda_coeff is not None:
    config["ssl"]["lambda_coeff"] = args.lambda_coeff

print("========== Training Config ==========")
print(f"YOLO weight/yaml : {args.weight}")
print(f"lambda_coeff    : {config['ssl']['lambda_coeff']}")
print("=====================================")

print(args)

train_dataset, val_dataset = build_datasets(config)

print(f"Train samples: {len(train_dataset)}")
print(f"Valid samples: {len(val_dataset)}")

train_loader = DataLoader(
    train_dataset,
    batch_size=config["train"]["batch_size"],
    shuffle=True,
    num_workers=config["train"]["num_workers"],
    pin_memory=True,
    drop_last=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=config["train"]["batch_size"],
    shuffle=False,
    num_workers=config["train"]["num_workers"],
    pin_memory=True,
    drop_last=True,
)

encoder = build_encoder(
    weight=args.weight
)

model = BarlowTwins(
    encoder=encoder,
    config=config,
    num_training_samples=len(train_dataset),
)

exp_name = build_exp_name(
    weight=args.weight,
    lambda_coeff=config["ssl"]["lambda_coeff"]
)
print("exp_name", exp_name)
checkpoint = ModelCheckpoint(
    dirpath=f"checkpoints/{exp_name}",
    save_top_k=1,
    save_last=True,
    every_n_epochs=1,
)

trainer = pl.Trainer(
    accelerator=config["system"]["accelerator"],
    devices=config["system"]["devices"],
    precision=config["train"]["precision"],
    max_epochs=config["train"]["max_epochs"],
    callbacks=[checkpoint],
    log_every_n_steps=100,
)

trainer.fit(
    model,
    train_dataloaders=train_loader,
    val_dataloaders=val_loader,
    ckpt_path=args.resume,
)

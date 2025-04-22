from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

from lightning.pytorch.loggers import WandbLogger
from pathlib import Path
from nntools.utils import Config

from dnafiber.data.dataset import FiberDatamodule
from dnafiber.trainee import Trainee
from dnafiber.callbacks import LogPredictionSamplesCallback
import torch
import argparse
from lightning import seed_everything

seed_everything(1234, workers=True)
torch.set_float32_matmul_precision("medium")


def train(arch, encoder):
    c = Config("configs/config.yaml")
    c["model"]["arch"] = arch
    c["model"]["encoder_name"] = encoder
    if "vit" in encoder:
        c["model"]["dynamic_img_size"] = True

    datamodule = FiberDatamodule(**c["data"])

    trainee = Trainee(c["training"], **c["model"])
    logger = WandbLogger(project="DeepFiberQ++", config=c.tracked_params)
    try:
        run_name = logger.experiment.name
        path = Path("checkpoints") / run_name
    except TypeError:
        path = Path("checkpoints") / "default"

    trainer = Trainer(
        **c["trainer"],
        callbacks=[
            LogPredictionSamplesCallback(logger),
            ModelCheckpoint(
                dirpath=path,
                monitor="jaccard",
                mode="max",
                save_last=True,
                save_top_k=1,
            ),
        ],
        logger=logger,
    )

    datamodule.setup()

    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()
    trainer.fit(
        trainee, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--arch", type=str, default="unet")
    argparser.add_argument("--encoder", type=str, default="resnet34")

    args = argparser.parse_args()
    arch = args.arch
    encoder = args.encoder
    print(f"Using {arch} with {encoder} encoder")
    train(arch=arch, encoder=encoder)

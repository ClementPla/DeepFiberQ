from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

from lightning.pytorch.loggers import WandbLogger
from pathlib import Path
from nntools.utils import Config

from dnafiber.data.dataset import FiberDatamodule
from dnafiber.trainee import Trainee, TraineeMaskRCNN
from dnafiber.callbacks import LogPredictionSamplesCallback
import torch
import argparse
from lightning import seed_everything
from dotenv import load_dotenv
import os
from huggingface_hub import HfApi
from lightning.pytorch.utilities import rank_zero_only

seed_everything(1234, workers=True)
torch.set_float32_matmul_precision("high")

load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN")


def train(arch, encoder):
    c = Config("configs/config.yaml")
    c["model"]["arch"] = arch
    c["model"]["encoder_name"] = encoder
    if "vit" in encoder:
        c["model"]["dynamic_img_size"] = True
    if "mit" in encoder:
        c["trainer"]["strategy"] = "ddp_find_unused_parameters_true"

    datamodule = FiberDatamodule(**c["data"], use_bbox=arch == "maskrcnn")
    traineeClass = TraineeMaskRCNN if arch == "maskrcnn" else Trainee

    trainee = traineeClass(**c["training"], **c["model"])

    logger = WandbLogger(
        project="DeepFiberQ++ Combined-Finetuned", config=c.tracked_params
    )
    try:
        run_name = logger.experiment.name
        path = Path("checkpoints") / "DeepFiberQ++ Combined-Finetuned" / run_name
    except TypeError:
        path = Path("checkpoints") / "DeepFiberQ++ Combined-Finetuned" / "default"

    callbacks = [
        ModelCheckpoint(
            dirpath=path,
            monitor="detection_recall",
            mode="max",
            save_last=True,
            save_top_k=1,
        ),
    ]
    trainer = Trainer(
        **c["trainer"],
        callbacks=callbacks,
        logger=logger,
    )

    datamodule.setup()

    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()
    trainer.fit(
        trainee, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )
    # model = traineeClass.load_from_checkpoint(
    #     "checkpoints/DeepFiberQ++ Combined-Finetuned/dauntless-dragon-7/epoch=699-step=8400.ckpt"
    # )

    upload_to_hub(trainee, arch, encoder)


@rank_zero_only
def upload_to_hub(model, arch, encoder):
    hfapi = HfApi()
    branch_name = f"{arch}_{encoder}_finetuned"
    hfapi.create_repo(
        "ClementP/DeepFiberQ",
        token=HF_TOKEN,
        exist_ok=True,
        repo_type="model",
    )
    hfapi.create_branch(
        "ClementP/DeepFiberQ",
        branch=branch_name,
        token=HF_TOKEN,
        exist_ok=True,
    )

    model.push_to_hub(
        "ClementP/DeepFiberQ",
        branch=branch_name,
        token=HF_TOKEN,
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

from lightning import Trainer
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    StochasticWeightAveraging,
)
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.loggers import WandbLogger
from pathlib import Path
from nntools.utils import Config
from lightning.pytorch.strategies import DDPStrategy
from dnafiber.data.dataset import FiberDatamodule
from dnafiber.trainee import Trainee, TraineeMaskRCNN
import torch
import argparse
from lightning import seed_everything
from dnafiber.model.utils import upload_to_hub


seed_everything(1234, workers=True)
torch.set_float32_matmul_precision("high")


def train(arch, encoder, use_swa=False):
    c = Config("configs/config.yaml")
    c["model"]["arch"] = arch
    c["model"]["encoder_name"] = encoder
    if "vit" in encoder:
        c["model"]["dynamic_img_size"] = True

    datamodule = FiberDatamodule(**c["data"], use_bbox=arch == "maskrcnn")
    traineeClass = TraineeMaskRCNN if arch == "maskrcnn" else Trainee
    datamodule.setup()

    trainee = traineeClass(**c["training"], **c["model"])

    logger = WandbLogger(project="DeepFiberQ++ V2", config=c.tracked_params)
    try:
        run_name = logger.experiment.name
        path = Path("checkpoints") / "DeepFiberQ++ V2" / run_name
    except TypeError:
        path = Path("checkpoints") / "DeepFiberQ++ V2" / "default"

    callbacks = [
        LearningRateMonitor(),
        ModelCheckpoint(
            dirpath=path,
            monitor="dice",
            mode="max",
            save_last=True,
            save_top_k=1,
        ),
        EarlyStopping(
            monitor="dice",
            mode="max",
            patience=10,
            verbose=True,
        ),
    ]

    if use_swa:
        callbacks.append(
            StochasticWeightAveraging(
                swa_lrs=1e-2,
            )
        )
    trainer = Trainer(
        **c["trainer"],
        callbacks=callbacks,
        logger=logger,
        sync_batchnorm=True,
        # fast_dev_run=2,
        strategy=DDPStrategy(find_unused_parameters=True),
    )

    tuner = Tuner(trainer=trainer)
    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()

    lr_finder = tuner.lr_find(
        model=trainee,
        max_lr=0.05,
        min_lr=1e-6,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        early_stop_threshold=None,
        num_training=50,
    )
    new_lr = lr_finder.suggestion()
    print(f"Suggested learning rate: {new_lr}")
    c["training"]["learning_rate"] = new_lr
    trainee.learning_rate = new_lr
    trainee.hparams.learning_rate = new_lr
    trainer.fit(
        model=trainee,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    # Load the best model from the checkpoint
    try:
        trainer.checkpoint_callback.best_model_path
    except AttributeError:
        print("No best model found, using the last checkpoint.")
        trainer.checkpoint_callback.best_model_path = (
            trainer.checkpoint_callback.last_model_path
        )

    trainee = traineeClass.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path,
        **c["training"],
        **c["model"],
    )

    upload_to_hub(model=trainee, arch=arch, encoder=encoder)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--arch", type=str, default="unet")
    argparser.add_argument("--encoder", type=str, default="resnet34")

    args = argparser.parse_args()
    arch = args.arch
    encoder = args.encoder
    print(f"Using {arch} with {encoder} encoder")
    train(arch=arch, encoder=encoder)

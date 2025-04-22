from lightning import LightningModule
import segmentation_models_pytorch as smp
from monai.losses.dice import GeneralizedDiceLoss
from monai.losses.cldice import SoftDiceclDiceLoss
from torchmetrics.classification import Dice, JaccardIndex
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics import MetricCollection
import torch.nn.functional as F


class Trainee(LightningModule):
    def __init__(self, training_config, **model_config):
        super().__init__()
        self.model = smp.create_model(classes=3, **model_config)
        self.loss = GeneralizedDiceLoss(to_onehot_y=False, softmax=False)
        self.diceclLoss = SoftDiceclDiceLoss()
        self.metric = MetricCollection(
            {
                "dice": Dice(num_classes=3, ignore_index=0),
                "jaccard": JaccardIndex(
                    num_classes=3, task="multiclass", ignore_index=0
                ),
            }
        )
        self.weight_decay = training_config["weight_decay"]
        self.learning_rate = training_config["learning_rate"]

    def forward(self, x):
        yhat = self.model(x)
        return yhat

    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["mask"]
        y_hat = self(x)
        loss = self.get_loss(y_hat, y)

        self.log("train_loss", loss)

        return loss

    def get_loss(self, y_hat, y):
        y_hat = F.softmax(y_hat, dim=1)
        y = F.one_hot(y.long(), num_classes=3)
        y = y.permute(0, 3, 1, 2).float()
        loss = self.diceclLoss(y_hat, y) + self.loss(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["image"], batch["mask"]
        y_hat = self(x)
        loss = self.get_loss(y_hat, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.metric.update(y_hat, y)
        self.log_dict(self.metric, on_step=False, on_epoch=True, sync_dist=True)
        return y_hat

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,  # type: ignore
            eta_min=self.learning_rate / 25,
        )
        scheduler = {
            "scheduler": scheduler,
            "interval": "epoch",
        }
        return [optimizer], [scheduler]

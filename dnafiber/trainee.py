from lightning import LightningModule
import segmentation_models_pytorch as smp
from monai.losses import GeneralizedDiceLoss, SoftDiceclDiceLoss
from torchmetrics.classification import Dice, JaccardIndex
from torch.optim import AdamW
from torchmetrics import MetricCollection
import torch.nn.functional as F


class Trainee(LightningModule):
    def __init__(self, training_config, **model_config):
        super().__init__()
        self.model = smp.create_model(classes=3, **model_config)
        self.loss = GeneralizedDiceLoss(to_onehot_y=True)
        self.diceclLoss = SoftDiceclDiceLoss()
        self.metric = MetricCollection(
            {
                "dice": Dice(num_classes=3),
                "jaccard": JaccardIndex(num_classes=3, task="multiclass"),
            }
        )
        self.weight_decay = training_config["weight_decay"]
        self.learning_rate = training_config["learning_rate"]

    def forward(self, x):
        yhat = self.model(x)
        return F.softmax(yhat, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["mask"]
        y_hat = self(x)
        loss = self.loss(y_hat, y.unsqueeze(1))

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["image"], batch["mask"]
        y_hat = self(x)
        loss = self.loss(y_hat, y.unsqueeze(1))
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.metric.update(y_hat, y)
        self.log_dict(self.metric, on_step=False, on_epoch=True, sync_dist=True)
        return y_hat

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        return optimizer

from lightning import LightningModule
import segmentation_models_pytorch as smp
from monai.losses.dice import GeneralizedDiceLoss
from torchmetrics.classification import Dice, JaccardIndex
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics import MetricCollection
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
import torch
import torchvision
from dnafiber.metric import DNAFIBERMetric
from dnafiber.model.steered_cnn import DNAFiberSteeredCNN
from skimage.measure import label


def _convert_activations(module, from_activation, to_activation):
    """Recursively convert activation functions in a module"""
    for name, child in module.named_children():
        if isinstance(child, from_activation):
            setattr(module, name, to_activation)
        else:
            _convert_activations(child, from_activation, to_activation)


class Trainee(LightningModule, PyTorchModelHubMixin):
    def __init__(
        self, learning_rate=0.001, weight_decay=0.0002, num_classes=3, **model_config
    ):
        super().__init__()
        self.model_config = model_config
        if (
            self.model_config.get("arch", None) is None
            or self.model_config["arch"] == "maskrcnn"
        ):
            self.model = None
        elif self.model_config["arch"] == "steered_cnn":
            self.model = DNAFiberSteeredCNN(**self.model_config)
        else:
            self.model = smp.create_model(classes=3, **self.model_config, dropout=0.2)
        self.loss = GeneralizedDiceLoss(to_onehot_y=False, softmax=False)
        self.metric = MetricCollection(
            {
                "dice": Dice(num_classes=num_classes, ignore_index=0),
                "jaccard": JaccardIndex(
                    num_classes=num_classes,
                    task="multiclass" if num_classes > 2 else "binary",
                    ignore_index=0,
                ),
                "detection": DNAFIBERMetric(),
            }
        )
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.save_hyperparameters()

    def forward(self, x):
        yhat = self.model(x)
        return yhat

    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["mask"]
        y = y.clamp(0, 2)
        y_hat = self(x)
        loss = self.get_loss(y_hat, y)

        self.log("train_loss", loss)

        return loss

    def get_loss(self, y_hat, y):
        y_hat = F.softmax(y_hat, dim=1)

        y = F.one_hot(y.long(), num_classes=3)
        y = y.permute(0, 3, 1, 2).float()
        loss = self.loss(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["image"], batch["mask"]
        y = y.clamp(0, 2)
        y_hat = self(x)
        loss = self.get_loss(y_hat, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.metric.update(y_hat, y)
        return y_hat

    def on_validation_epoch_end(self):
        scores = self.metric.compute()
        self.log_dict(scores, sync_dist=True)
        self.metric.reset()

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

    def get_fiber_probability(self, probas: torch.Tensor):
        pos_pred = 1 - probas[:, 0, :, :]
        preds = pos_pred > 1 / 2
        binary = preds.long().detach().cpu()
        labelmap = torch.zeros_like(binary, dtype=torch.long)
        for i, p in enumerate(binary):
            labelmap[p] = torch.from_numpy(label(p.numpy(), connectivity=2))
        labelmap = labelmap.to(probas.device)

        return labelmap, pos_pred


class TraineeMaskRCNN(Trainee):
    def __init__(self, learning_rate=0.001, weight_decay=0.0002, **model_config):
        super().__init__(learning_rate, weight_decay, **model_config)
        self.model = torchvision.models.get_model("maskrcnn_resnet50_fpn_v2")

    def forward(self, x):
        yhat = self.model(x)
        return yhat

    def training_step(self, batch, batch_idx):
        image = batch["image"]
        targets = batch["targets"]
        loss_dict = self.model(image, targets)
        losses = sum(loss for loss in loss_dict.values())
        self.log("train_loss", losses, on_step=True, on_epoch=False, sync_dist=True)
        return losses

    def validation_step(self, batch, batch_idx):
        image = batch["image"]
        targets = batch["targets"]

        predictions = self.model(image)
        b = len(predictions)
        predicted_masks = []
        gt_masks = []
        for i in range(b):
            scores = predictions[i]["scores"]
            masks = predictions[i]["masks"]
            good_masks = masks[scores > 0.5]
            # Combined into a single mask
            good_masks = torch.sum(good_masks, dim=0)
            predicted_masks.append(good_masks)
            gt_masks.append(targets[i]["masks"].sum(dim=0))

        gt_masks = torch.stack(gt_masks).squeeze(1) > 0
        predicted_masks = torch.stack(predicted_masks).squeeze(1) > 0
        self.metric.update(predicted_masks, gt_masks)
        return predictions

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


from lightning.pytorch.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only
import wandb

class LogPredictionSamplesCallback(Callback):
    def __init__(self, wandb_logger, n_images=8):
        self.n_images = n_images
        self.wandb_logger = wandb_logger
        super().__init__()

    @rank_zero_only
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if (
            batch_idx < 1
            and trainer.is_global_zero
        ):
            n = self.n_images
            x = batch["image"][:n].float()
            h, w = x.shape[-2:]
            y = batch["mask"][:n]
            pred = outputs[:n]
            pred = pred.argmax(dim=1)

            if len(y.shape) == 4:
                y = y.squeeze(1)
            if len(pred.shape) == 4:
                pred = pred.squeeze(1)

            columns = ["image"]
            class_labels = {0: "Background", 1: "Red", 2: "Green"}

            data = [
                [
                    wandb.Image(
                        x_i,
                        masks={
                            "Prediction": {
                                "mask_data": p_i.cpu().numpy(),
                                "class_labels": class_labels,
                            },
                            "Groundtruth": {
                                "mask_data": y_i.cpu().numpy(),
                                "class_labels": class_labels,
                            },
                        },
                    )
                ]
                for x_i, y_i, p_i in list(zip(x, y, pred))
            ]
            self.wandb_logger.log_table(
                data=data, key=f"Validation Batch {batch_idx}", columns=columns
            )

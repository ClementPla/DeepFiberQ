import kornia as K
import torch
import torchmetrics.functional as F
from skimage.measure import label
from torchmetrics import Metric


class DNAFIBERMetric(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.add_state(
            "detection_tp",
            default=torch.tensor(0, dtype=torch.int64),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "fiber_red_dice",
            default=torch.tensor(0, dtype=torch.float32),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "fiber_green_dice",
            default=torch.tensor(0, dtype=torch.float32),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "fiber_red_recall",
            default=torch.tensor(0, dtype=torch.float32),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "fiber_green_recall",
            default=torch.tensor(0, dtype=torch.float32),
            dist_reduce_fx="sum",
        )
        # Specificity
        self.add_state(
            "fiber_red_precision",
            default=torch.tensor(0, dtype=torch.float32),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "fiber_green_precision",
            default=torch.tensor(0, dtype=torch.float32),
            dist_reduce_fx="sum",
        )

        self.add_state(
            "detection_fp",
            default=torch.tensor(0, dtype=torch.int64),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "N",
            default=torch.tensor(0, dtype=torch.int64),
            dist_reduce_fx="sum",
        )

        self.add_state(
            "N_predicted",
            default=torch.tensor(0, dtype=torch.int64),
            dist_reduce_fx="sum",
        )

    def update(self, preds, target):
        if preds.ndim == 4:
            preds = preds.argmax(dim=1)
        if target.ndim == 4:
            target = target.squeeze(1)
        B, H, W = preds.shape
        preds_labels = []
        target_labels = []
        binary_preds = preds > 0
        binary_target = target > 0
        N_true_labels = 0
        for i in range(B):
            pred = binary_preds[i].detach().cpu().numpy()
            target_np = binary_target[i].detach().cpu().numpy()
            pred_labels = label(pred, connectivity=2)
            target_labels_np = label(target_np, connectivity=2)
            preds_labels.append(torch.from_numpy(pred_labels).to(preds.device))
            target_labels.append(torch.from_numpy(target_labels_np).to(preds.device))
            N_true_labels += target_labels_np.max()
            self.N_predicted += pred_labels.max()

        preds_labels = torch.stack(preds_labels)
        target_labels = torch.stack(target_labels)

        for i, plab in enumerate(preds_labels):
            labels = torch.unique(plab)
            for blob in labels:
                if blob == 0:
                    continue
                pred_mask = plab == blob
                pixels_in_common = torch.any(pred_mask & binary_target[i])
                if pixels_in_common:
                    self.detection_tp += 1
                    gt_label = target_labels[i][pred_mask].unique()[-1]
                    gt_mask = target_labels[i] == gt_label
                    common_mask = pred_mask | gt_mask
                    pred_fiber = preds[i][common_mask]
                    gt_fiber = target[i][common_mask]
                    dices = F.dice(
                        pred_fiber,
                        gt_fiber,
                        num_classes=3,
                        ignore_index=0,
                        average=None,
                    )
                    dices = torch.nan_to_num(dices, nan=0.0)
                    self.fiber_red_dice += dices[1]
                    self.fiber_green_dice += dices[2]
                    recalls = F.recall(
                        pred_fiber,
                        gt_fiber,
                        num_classes=3,
                        ignore_index=0,
                        task="multiclass",
                        average=None,
                    )
                    recalls = torch.nan_to_num(recalls, nan=0.0)
                    self.fiber_red_recall += recalls[1]
                    self.fiber_green_recall += recalls[2]

                    # Specificity
                    precision = F.precision(
                        pred_fiber,
                        gt_fiber,
                        num_classes=3,
                        ignore_index=0,
                        task="multiclass",
                        average=None,
                    )
                    precision = torch.nan_to_num(precision, nan=0.0)
                    self.fiber_red_precision += precision[1]
                    self.fiber_green_precision += precision[2]

                else:
                    self.detection_fp += 1

        self.N += N_true_labels

    def compute(self):
        return {
            "detection_precision": self.detection_tp
            / (self.detection_tp + self.detection_fp + 1e-7),
            "detection_recall": self.detection_tp / (self.N + 1e-7),
            "fiber_red_dice": self.fiber_red_dice / (self.detection_tp + 1e-7),
            "fiber_green_dice": self.fiber_green_dice / (self.detection_tp + 1e-7),
            "fiber_red_recall": self.fiber_red_recall / (self.detection_tp + 1e-7),
            "fiber_green_recall": self.fiber_green_recall / (self.detection_tp + 1e-7),
            "fiber_red_precision": self.fiber_red_precision
            / (self.detection_tp + 1e-7),
            "fiber_green_precision": self.fiber_green_precision
            / (self.detection_tp + 1e-7),
            "total_real_fibers": self.N.item(),
            "total_predicted_fibers": self.N_predicted.item(),
        }

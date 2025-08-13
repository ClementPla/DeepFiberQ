
import torch
from dnafiber.correction.data import DNADataset
from pathlib import Path
import timm
from tqdm.auto import tqdm
from torchmetrics import Accuracy, F1Score, CohenKappa, Precision, Recall, Specificity, MetricCollection, AUROC

import wandb

def train_one_epoch(model, dataloader, optimizer, criterion, device, current_epoch=0, concat_pred=True):
    model.train()
    total_loss = 0.0
    for imgs, preds, labels in tqdm(dataloader, desc=f"Epoch {current_epoch+1}"):
        imgs = imgs.to(device)
        preds = preds.to(device).unsqueeze(1)
        if concat_pred:
            imgs = torch.cat((imgs, preds), dim=1)  # Concatenate images and predictions along the channel dimension
        labels = labels.to(device).unsqueeze(1)  # Ensure labels are of shape (batch_size, 1)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, metrics, device, concat_pred=True):
    model.eval()
    metrics.reset()
    with torch.no_grad():
        for imgs, preds, labels in dataloader:
            imgs = imgs.to(device)
            preds = preds.to(device)
            if concat_pred:
                imgs = torch.cat((imgs, preds.unsqueeze(1)), dim=1)
            labels = labels.to(device)
            outputs = model(imgs).sigmoid()
            metrics.update(outputs, labels.unsqueeze(1))
    return metrics.compute()


def main():
    
    root_imgs = Path("/home/clement/Documents/data/DNAFiber/DNAICorrection/Input/")
    root_prediction = Path("/home/clement/Documents/data/DNAFiber/DNAICorrection/Output/DNAICorrection/export/multiclass/local/")
    root_df = Path("/home/clement/Documents/data/DNAFiber/DNAICorrection/classification.csv")

    root_imgs = Path("/home/clement/Documents/data/DNAFiber/DNAICorrection/Input/")
    root_prediction = Path(
        "/home/clement/Documents/data/DNAFiber/DNAICorrection/Output/DNAICorrection/export/multiclass/local/"
    )
    root_df = Path(
        "/home/clement/Documents/data/DNAFiber/DNAICorrection/classification.csv"
    )

    dataset = DNADataset(
        root_images=root_imgs, root_predictions=root_prediction, root_csv=root_df
    )
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(42))
    test_dataset.dataset.is_train = False  # Set test dataset to not use augmentations
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=128, shuffle=False, num_workers=4,
        pin_memory=True
    )

    

    metrics = MetricCollection(
    [
        Accuracy(num_classes=2, task="binary"),
        F1Score(num_classes=2, task="binary"),
        CohenKappa(num_classes=2, task="binary"), 
        Precision(num_classes=2, task="binary"),
        Recall(num_classes=2, task="binary"),
        Specificity(num_classes=2, task="binary"),
        AUROC(num_classes=2, task="binary")
    ]
)
    with wandb.init(project="DNA-Fiber-Correction") as run:

        model_name = run.config["model"]
        lr = run.config["lr"]
        wc = run.config["wc"]
        pc = run.config["pc"]
        concat_pred = run.config.get("concat_pred", True)

        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pc], device="cuda" if torch.cuda.is_available() else "cpu"))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        N_EPOCHS = 50    
        try:
            model = timm.create_model(model_name, pretrained=True, num_classes=1, in_chans=4 if concat_pred else 3, img_size=128)
        except Exception as e:
            model = timm.create_model(model_name, pretrained=True, num_classes=1, in_chans=4 if concat_pred else 3)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wc)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS, eta_min=1e-6)
        model.train()
        model.to(device)
        metrics.to(device)
        for epoch in range(N_EPOCHS):
            train_loss = train_one_epoch(model, train_dataloader, optimizer, criterion, device, current_epoch=epoch, concat_pred=concat_pred)
            metrics_result = evaluate(model, test_dataloader, metrics, device, concat_pred=concat_pred)
            wandb.log({
                    "train_loss": train_loss,
                    "val_acc": metrics_result["BinaryAccuracy"].item(),
                    "val_f1": metrics_result["BinaryF1Score"].item(),
                    "val_kappa": metrics_result["BinaryCohenKappa"].item(),
                    "val_precision": metrics_result["BinaryPrecision"].item(),
                    "val_recall": metrics_result["BinaryRecall"].item(),
                    "val_specificity": metrics_result["BinarySpecificity"].item(),
                })
            scheduler.step()


if __name__=="__main__":
    # Define a sweep config dictionary
    sweep_configuration = {
        "method": "bayes",
        "name": "DNA-Fiber-Correction-Sweep",
        "metric": {"goal": "maximize", "name": "val_kappa"},
        "parameters": {
            "model": {"values": ["resnet18", "seresnet50.a1_in1k", "efficientnet_b0", "mobilenetv3_small_100", "efficientnet_b2", 
                                 "vit_base_patch16_224"]},
            "wc": {"max": 0.01, "min": 0.0001},
            "pc": {"max": 10.0, "min": 1.0},
            "lr": {"max": 0.01, "min": 0.0001},
            "concat_pred": {"values": [True, False]}
        },
        "early_terminate": {
            "type": "hyperband",
            "min_iter": 5,
            "max_iter": 100,
            "eta": 3
        }

    }

    sweep_id = wandb.sweep(sweep=sweep_configuration, project="DNA-Fiber-Correction")

    wandb.agent(sweep_id, function=main)
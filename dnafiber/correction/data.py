import pandas as pd
from torch.utils.data import Dataset
import numpy as np
import cv2
import albumentations as A
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class DNADataset(Dataset):
    def __init__(self, root_csv, root_images, root_predictions):
        self.root_csv = root_csv
        self.root_images = root_images
        self.root_predictions = root_predictions

        self.df = pd.read_csv(self.root_csv)
        self.prepare_df(self.df)
        self.is_train = True  # Set to False for test dataset

        filenames = self.df["filename"].unique()
        self.imgs = []
        self.predictions = []
        for filename in filenames:
            if not (self.root_images / filename).exists():
                print(f"Missing image: {filename}")
            if not (self.root_predictions / filename).exists():
                print(f"Missing prediction: {filename}")

        self.imgs = [cv2.imread(str(self.root_images / filename))[:,:, ::-1] for filename in filenames if (self.root_images / filename).exists()]
        self.predictions = [cv2.imread(str(self.root_predictions / filename), cv2.IMREAD_GRAYSCALE) for filename in filenames if (self.root_predictions / filename).exists()]
        self.train_transform = A.Compose([
            A.Compose([
                A.Affine(scale=(0.5, 1.5), translate_percent=(-0.1, 0.1), border_mode=cv2.BORDER_REFLECT),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),

                # A.GaussianBlur(blur_limit=(3, 7), p=0.5),
                # A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=20, val_shift_limit=20, p=0.5),
                # A.RandomGamma(gamma_limit=(80, 120), p=0.5),
            ]),

            
        ])
        self.test_transform = A.Compose([A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            A.pytorch.ToTensorV2()])
    def prepare_df(self, df):

        df["filename"] = df["filename"].apply(lambda x: x.split("(")[0] + ".png").apply(lambda x: x.replace(".png.png", ".png"))

        duplicate_filenames = df["filename"][df["filename"].duplicated()].unique()
        # For each duplicate filename, keep the occurence with a non-nan "Quality" value and drop the others
        for filename in duplicate_filenames:
            # Suppress the FutureWarning by separating .ffill/.bfill and .infer_objects
            quality_filled = df.loc[df["filename"] == filename, "Quality"].infer_objects(copy=False).ffill().bfill()
            df.loc[df["filename"] == filename, "Quality"] = quality_filled
        df.drop_duplicates(subset=["filename"], keep="first", inplace=True)
        df.replace({"Quality" : {pd.NA: 0.0, "Bad": 1.0, "Good":0.0}}, inplace=True)

    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img = self.imgs[idx]
        prediction = self.predictions[idx]
        quality = self.df["Quality"].iloc[idx]

        if img is None or prediction is None:
            raise ValueError(f"Image or prediction at index {idx} is None")
        if self.is_train:
            augmented = self.train_transform(image=img, mask=prediction)
            img = augmented["image"]
            prediction = augmented["mask"]
        
        augmented = self.test_transform(image=img, mask=prediction)
        img_transformed = augmented["image"]
        pred_transformed = augmented["mask"]

        return img_transformed, pred_transformed, quality
    
    def plot(self, idx=None):
        if idx is None:
            idx = np.random.randint(0, len(self))
        if idx < 0:
            idx = len(self) + idx
        if idx >= len(self):
            idx = idx % len(self)
            
        img, pred, quality = self[idx]
        print(pred.shape)
        # Normalize the image between 0 and 1 for visualization
        img = (img - img.min()) / (img.max() - img.min())
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(img.permute(1, 2, 0).numpy())
        ax[0].set_title(f"Image - Quality: {"Bad" if quality==1.0 else "Good"}")
        ax[1].imshow(pred.numpy(), cmap=ListedColormap(['black', 'red', 'green']))
        ax[1].set_title("Prediction")
        plt.show()
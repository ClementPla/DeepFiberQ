import albumentations as A
import nntools.dataset as D
import numpy as np
from albumentations.pytorch import ToTensorV2
from lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


@D.nntools_wrapper
def convert_mask(mask):
    output = np.zeros(mask.shape[:2], dtype=np.uint8)
    output[mask[:, :, 0] > 200] = 1
    output[mask[:, :, 1] > 200] = 2
    return {"mask": output}


class FiberDatamodule(LightningDataModule):
    def __init__(
        self,
        root_img,
        crop_size=(256, 256),
        batch_size=32,
        num_workers=8,
        **kwargs,
    ):
        self.root_img = str(root_img)
        self.crop_size = crop_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.kwargs = kwargs

        super().__init__()

    def setup(self, *args, **kwargs):
        def _get_dataset():
            dataset = D.MultiImageDataset(
                {
                    "image": f"{self.root_img}/images/",
                    "mask": f"{self.root_img}/annotations/",
                },
                shape=(1024, 1024),
            )  # type: ignore
            dataset.img_filepath["image"] = np.asarray(  # type: ignore
                sorted(
                    list(dataset.img_filepath["image"]),
                    key=lambda x: (x.parent.stem, x.stem),
                )
            )
            dataset.img_filepath["mask"] = np.asarray(  # type: ignore
                sorted(
                    list(dataset.img_filepath["mask"]),
                    key=lambda x: (x.parent.stem, x.stem),
                )
            )
            dataset.composer = D.Composition()
            dataset.composer << convert_mask  # type: ignore

            return dataset

        self.train = _get_dataset()
        self.val = _get_dataset()
        stratify = [int(f.parent.stem) for f in self.train.img_filepath["image"]]
        train_idx, val_idx = train_test_split(
            np.arange(len(self.train)),  # type: ignore
            stratify=stratify,
            test_size=0.2,
            random_state=42,
        )
        self.train.subset(train_idx)
        self.val.subset(val_idx)

        self.train.composer.add(*self.get_train_composer())
        self.val.composer.add(*self.cast_operators())

    def get_train_composer(self):
        return [
            A.Compose(
                [
                    A.CropNonEmptyMaskIfExists(
                        width=self.crop_size[0], height=self.crop_size[1]
                    ),
                    A.HorizontalFlip(),
                    A.VerticalFlip(),
                    A.Affine(),
                    A.ElasticTransform(),
                    A.RandomRotate90(),
                    A.OneOf(
                        [
                            A.RandomBrightnessContrast(
                                brightness_limit=(-0.2, 0.1),
                                contrast_limit=(-0.2, 0.1),
                                p=0.5,
                            ),
                            A.HueSaturationValue(
                                hue_shift_limit=(-5, 5),
                                sat_shift_limit=(-20, 20),
                                val_shift_limit=(-20, 20),
                                p=0.5,
                            ),
                        ]
                    ),
                    A.GaussNoise(std_range=(0.0, 0.1), p=0.5),
                ]
            ),
            *self.cast_operators(),
        ]

    def cast_operators(self):
        return [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

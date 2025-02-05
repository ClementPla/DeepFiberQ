from nntools.dataset.image_dataset import ImageDataset
from pathlib import Path
import json
import cv2
from lightning import LightningDataModule
from nntools.dataset import Composition
import albumentations as A
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2
from dnafiber.data.utils import read_svg
from skimage.segmentation import expand_labels


class FiberDataset(ImageDataset):
    def __init__(
        self,
        root_img,
        revision_file,
        root_svg,
        train,
        **kwargs,
    ):
        revisions = json.load(open(revision_file))
        self.revisions = list(set(revisions["images"]))
        self.root_svg = root_svg
        self.train = train
        super().__init__(img_root=root_img, **kwargs)

    def list_files(self, recursive):
        self.all_imgs = {}
        self.all_masks = {}
        img_paths = list(Path(self.img_root[0]).glob("*.png"))
        train_paths, test_paths = train_test_split(
            img_paths, test_size=0.2, random_state=1234
        )
        if self.train:
            img_paths = train_paths
        else:
            img_paths = test_paths
        for img in img_paths:
            img = Path(img)
            if img.stem + ".png" in self.revisions:
                img_array = cv2.imread(str(img))[:, :, ::-1]
                svg_path = Path(self.root_svg) / (img.stem + ".svg")
                mask = read_svg(str(svg_path))
                self.all_imgs[img.stem] = img_array
                self.all_masks[img.stem] = mask

        self.img_filepath["image"] = [
            Path(self.img_root[0]) / (img + ".png") for img in self.all_imgs.keys()
        ]

    def read_from_disk(self, item):
        img_name = self.filename(item).split(".png")[0]
        img = self.all_imgs[img_name]
        mask = self.all_masks[img_name]
        img = self.resize_and_pad(image=img, interpolation=self.interpolation_flag)
        mask = self.resize_and_pad(mask, interpolation=cv2.INTER_NEAREST_EXACT)
        # mask = expand_labels(mask, distance=5)
        return {"image": img, "mask": mask}


class FiberDatamodule(LightningDataModule):
    def __init__(
        self,
        root_img,
        root_svg,
        revision_file,
        crop_size=(256, 256),
        batch_size=32,
        num_workers=8,
        **kwargs,
    ):
        self.root_img = str(root_img)
        self.root_svg = str(root_svg)
        self.revision_file = str(revision_file)
        self.crop_size = crop_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.kwargs = kwargs

        super().__init__()

    def setup(self, *args, **kwargs):
        self.train = FiberDataset(
            root_img=self.root_img,
            revision_file=self.revision_file,
            root_svg=self.root_svg,
            train=True,
            **self.kwargs,
        )
        self.val = FiberDataset(
            root_img=self.root_img,
            revision_file=self.revision_file,
            root_svg=self.root_svg,
            train=False,
            **self.kwargs,
        )

        self.train.composer = self.get_train_composer()
        self.val.composer = Composition()
        self.val.composer.add(*self.cast_operators())

    def get_train_composer(self):
        c = Composition()
        c << A.Compose(
            [
                A.CropNonEmptyMaskIfExists(
                    width=self.crop_size[0], height=self.crop_size[1]
                ),
                A.HorizontalFlip(),
                A.VerticalFlip(),
                A.Affine(),
                A.ElasticTransform(),
                A.RandomRotate90(),
            ]
        )
        c.add(*self.cast_operators())
        return c

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
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

import albumentations as A
import nntools.dataset as D
import numpy as np
from albumentations.pytorch import ToTensorV2
from lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize, dilation
from skimage.segmentation import expand_labels
import torch
from nntools.dataset.composer import CacheBullet
from skimage.filters import sato
import cv2
from timm.data.loader import MultiEpochsDataLoader

cv2.setNumThreads(0)


@D.nntools_wrapper
def to_polar_space(image, mask):
    polar_image = cv2.linearPolar(
        image,
        center=(image.shape[1] // 2, image.shape[0] // 2),
        maxRadius=image.shape[0] // 2,
        flags=cv2.WARP_FILL_OUTLIERS | cv2.INTER_LINEAR,
    )

    polar_mask = cv2.linearPolar(
        mask,
        center=(mask.shape[1] // 2, mask.shape[0] // 2),
        maxRadius=mask.shape[0] // 2,
        flags=cv2.WARP_FILL_OUTLIERS | cv2.INTER_NEAREST,
    )
    return {"image": polar_image, "mask": polar_mask}


@D.nntools_wrapper
def convert_mask(mask):
    output = np.zeros(mask.shape[:2], dtype=np.uint8)
    output[mask[:, :, 0] > 150] = 1
    output[mask[:, :, 1] > 150] = 2
    binary_mask = output > 0
    skeleton = skeletonize(binary_mask) * output
    output = expand_labels(skeleton, 3)
    output = np.clip(output, 0, 2)
    output = output.astype(np.uint8)
    return {"mask": output}


@D.nntools_wrapper
def sato_filter(image):
    red_channel = image[:, :, 0]
    green_channel = image[:, :, 1]
    sigmas = np.arange(1, 2, 0.5)
    red_channel = sato(red_channel, black_ridges=False, sigmas=sigmas).astype(np.uint8)
    green_channel = sato(green_channel, black_ridges=False, sigmas=sigmas).astype(
        np.uint8
    )

    image = np.stack(
        [red_channel, green_channel, np.zeros_like(green_channel)], axis=-1
    )
    return {
        "image": image,
    }


@D.nntools_wrapper
def extract_bbox(mask):
    binary_mask = mask > 0
    labelled = label(binary_mask)
    props = regionprops(labelled, intensity_image=mask)
    skeleton = skeletonize(binary_mask) * mask
    mask = dilation(skeleton, np.ones((3, 3)))
    bboxes = []
    masks = []
    # We want the XYXY format
    for prop in props:
        minr, minc, maxr, maxc = prop.bbox
        bboxes.append([minc, minr, maxc, maxr])
        masks.append((labelled == prop.label).astype(np.uint8))
    if not masks:
        masks = np.zeros_like(mask)[np.newaxis, :, :]
    masks = np.array(masks)
    masks = np.moveaxis(masks, 0, -1)

    return {
        "bboxes": np.array(bboxes),
        "mask": masks,
        "fiber_ids": np.array([p.label for p in props]),
    }


class FiberDatamodule(LightningDataModule):
    def __init__(
        self,
        root_img,
        crop_size=None,
        shape=1024,
        batch_size=32,
        num_workers=8,
        use_bbox=False,
        sato_filter=False,
        polar_space=False,
        **kwargs,
    ):
        self.shape = shape
        self.root_img = str(root_img)
        self.crop_size = crop_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.kwargs = kwargs
        self.use_bbox = use_bbox
        self.sato_filter = sato_filter
        self.polar_space = polar_space

        super().__init__()

    def setup(self, *args, **kwargs):
        def _get_dataset(version):
            dataset = D.MultiImageDataset(
                {
                    "image": f"{self.root_img}/{version}/images/",
                    "mask": f"{self.root_img}/{version}/annotations/",
                },
                shape=(self.shape, self.shape),
                use_cache=self.kwargs.get("use_cache", False),
                cache_option=self.kwargs.get("cache_option", None),
                id=version,
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
            if self.use_bbox:
                dataset.composer << extract_bbox

            return dataset

        self.train = _get_dataset("train")
        self.val = _get_dataset("train")
        self.test = _get_dataset("test")
        self.train.composer << CacheBullet()
        self.val.use_cache = False
        self.test.use_cache = False

        stratify = []
        for f in self.train.img_filepath["mask"]:
            if "tile" in f.stem:
                stratify.append(int(f.parent.stem))
            else:
                stratify.append(25)
        train_idx, val_idx = train_test_split(
            np.arange(len(self.train)),  # type: ignore
            stratify=stratify,
            test_size=0.1,
            random_state=42,
        )
        self.train.subset(train_idx)
        self.val.subset(val_idx)

        self.train.composer.add(*self.get_train_composer())
        self.val.composer.add(*self.cast_operators())
        self.test.composer.add(*self.cast_operators())

    def get_train_composer(self):
        transforms = []
        if self.crop_size is not None:
            transforms.append(
                A.CropNonEmptyMaskIfExists(
                    width=self.crop_size[0], height=self.crop_size[1]
                ),
            )
        return [
            A.Compose(
                transforms
                + [
                    A.Affine(
                        scale=(0.5, 2),
                        rotate=(-75, 75),
                        p=0.75,
                        border_mode=cv2.BORDER_REFLECT101,
                        balanced_scale=True,
                        keep_ratio=True,
                        mask_interpolation=cv2.INTER_NEAREST,
                    ),
                ]
                + [
                    A.SquareSymmetry(
                        p=0.5,
                    ),
                    A.OneOf(
                        [
                            A.RandomBrightnessContrast(
                                brightness_limit=(-0.2, 0.2),
                                contrast_limit=(-0.2, 0.2),
                                p=0.75,
                            ),
                            A.HueSaturationValue(
                                hue_shift_limit=(-10, 10),
                                sat_shift_limit=(-20, 20),
                                val_shift_limit=(-20, 20),
                                p=0.75,
                            ),
                        ],
                        p=0.75,
                    ),
                    A.GaussNoise(std_range=(0.0, 0.05), p=0.2),
                ],
                bbox_params=A.BboxParams(
                    format="pascal_voc", label_fields=["fiber_ids"], min_visibility=0.95
                )
                if self.use_bbox
                else None,
            ),
            *self.cast_operators(),
        ]

    def cast_operators(self):
        return [
            A.Normalize(max_pixel_value=255.0),
            ToTensorV2(),
        ]

    def train_dataloader(self):
        if self.use_bbox:
            return DataLoader(
                self.train,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                persistent_workers=True,
                collate_fn=bbox_collate_fn,
            )

        else:
            return DataLoader(
                self.train,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                persistent_workers=True,
            )

    def val_dataloader(self):
        if self.use_bbox:
            return DataLoader(
                self.val,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                persistent_workers=True,
                collate_fn=bbox_collate_fn,
            )
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        if self.use_bbox:
            return DataLoader(
                self.test,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                persistent_workers=True,
                collate_fn=bbox_collate_fn,
            )
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=1,
            persistent_workers=False,
        )


def bbox_collate_fn(batch):
    images = []
    targets = []

    for b in batch:
        target = dict()

        target["boxes"] = torch.from_numpy(b["bboxes"])
        if target["boxes"].shape[0] == 0:
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
        images.append(b["image"])
        target["boxes"] = torch.from_numpy(b["bboxes"])
        target["masks"] = b["mask"].permute(2, 0, 1)
        if target["boxes"].shape[0] == 0:
            target["labels"] = torch.zeros(1, dtype=torch.int64)
        else:
            target["labels"] = torch.ones_like(target["boxes"][:, 0], dtype=torch.int64)

        targets.append(target)

    return {
        "image": torch.stack(images),
        "targets": targets,
    }

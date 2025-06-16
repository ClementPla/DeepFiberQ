import torch.nn.functional as F
import numpy as np
import torch
from torchvision.transforms._functional_tensor import normalize
import pandas as pd
from skimage.segmentation import expand_labels
from skimage.measure import label
import albumentations as A
from monai.inferers import SlidingWindowInferer
from dnafiber.deployment import _get_model
from dnafiber.postprocess import refine_segmentation

transform = A.Compose(
    [
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.ToTensorV2(),
    ]
)


def preprocess_image(image):
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    image = normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return image


def convert_to_dataset(counts):
    data = {"index": [], "red": [], "green": [], "ratio": []}
    for k, v in counts.items():
        data["index"].append(k)
        data["green"].append(v["green"])
        data["red"].append(v["red"])
        if v["red"] == 0:
            data["ratio"].append(np.nan)
        else:
            data["ratio"].append(v["green"] / (v["red"]))
    df = pd.DataFrame(data)
    return df


def convert_mask_to_image(mask, expand=False):
    if expand:
        mask = expand_labels(mask, distance=expand)
    h, w = mask.shape
    image = np.zeros((h, w, 3), dtype=np.uint8)
    GREEN = np.array([0, 255, 0])
    RED = np.array([255, 0, 0])

    image[mask == 1] = RED
    image[mask == 2] = GREEN

    return image


@torch.inference_mode()
def infer(model, image, device, scale=0.13, to_numpy=True, only_probabilities=False):
    if isinstance(model, str):
        model = _get_model(device=device, revision=model)
    model_pixel_size = 0.26

    scale = scale / model_pixel_size
    tensor = transform(image=image)["image"].unsqueeze(0).to(device)
    h, w = tensor.shape[2], tensor.shape[3]
    device = torch.device(device)
    with torch.autocast(device_type=device.type):
        tensor = F.interpolate(
            tensor,
            size=(int(h * scale), int(w * scale)),
            mode="bilinear",
        )
        if tensor.shape[2] > 1024 or tensor.shape[3] > 1024:
            inferer = SlidingWindowInferer(
                roi_size=(1024, 1024),
                sw_batch_size=4,
                overlap=0.25,
                mode="gaussian",
                device=device,
                progress=True,
            )
            output = inferer(tensor, model)
        else:
            output = model(tensor)

        probabilities = F.softmax(output, dim=1)
        if only_probabilities:
            probabilities = probabilities.cpu()

            probabilities = F.interpolate(
                probabilities,
                size=(h, w),
                mode="bilinear",
            )
            return probabilities

        output = F.interpolate(
            probabilities.argmax(dim=1, keepdim=True).float(),
            size=(h, w),
            mode="nearest",
        )

    output = output.squeeze().byte()
    if to_numpy:
        output = output.cpu().numpy()

    return output

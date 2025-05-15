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
from dnafiber.postprocess.morphology import get_clean_skeleton_gpu

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


def post_process(prediction_map, extract_properties=False):
    prediction_map = get_clean_skeleton_gpu(prediction_map) * prediction_map
    # prediction_map = skeletonize(prediction_map > 0, method="lee") * prediction_map
    type_map = np.zeros_like(prediction_map)
    labeled_mask, count = label(prediction_map > 0, return_num=True, connectivity=2)
    if extract_properties:
        properties = dict()
    for i in range(1, count + 1):
        current_fiber = labeled_mask == i
        bbox = np.where(current_fiber)
        minr, minc, maxr, maxc = (
            bbox[0].min(),
            bbox[1].min(),
            bbox[0].max(),
            bbox[1].max(),
        )
        patch_fiber = (
            current_fiber[minr:maxr, minc:maxc] * prediction_map[minr:maxr, minc:maxc]
        )
        # Check if the fiber is too small or too large

        if (patch_fiber > 0).sum() < 20:
            prediction_map[current_fiber] = 0
            continue
        _, n_segments = label(patch_fiber, return_num=True, connectivity=2)

        if n_segments == 1 or n_segments > 3:
            prediction_map[current_fiber] = 0
            continue
        type_map[current_fiber] = n_segments

        if extract_properties:
            properties[i] = {
                "red": np.sum(patch_fiber == 1),
                "green": np.sum(patch_fiber == 2),
                "type": "bilateral" if n_segments == 2 else "trilateral",
                "bbox": {
                    "minr": minr,
                    "minc": minc,
                    "maxr": maxr,
                    "maxc": maxc,
                },
            }

    prediction_map = np.stack([prediction_map, type_map], axis=-1)

    # prediction_map = expand_labels(prediction_map, distance=5)
    if extract_properties:
        return prediction_map, properties
    else:
        return prediction_map


def infer(model, image, device, scale=0.13):
    if isinstance(model, str):
        model = _get_model(device=device, revision=model)
    model_pixel_size = 0.26

    scale = scale / model_pixel_size
    tensor = transform(image=image)["image"].unsqueeze(0).to(device)
    h, w = tensor.shape[2], tensor.shape[3]
    tensor = F.interpolate(
        tensor,
        size=(int(h * scale), int(w * scale)),
        mode="bilinear",
    )
    inferer = SlidingWindowInferer(
        roi_size=(2048, 2048),
        sw_batch_size=1,
        overlap=0.25,
        mode="gaussian",
        device=device,
        progress=True,
    )
    with torch.inference_mode():
        output = inferer(tensor, model)
        output = F.interpolate(
            output.argmax(dim=1, keepdim=True).float(),
            size=(h, w),
            mode="nearest",
        )
        output = output.squeeze().long().cpu().numpy()

    return output

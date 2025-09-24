import math
import time
from enum import Enum

import cv2
import pandas as pd
import torch
from joblib import Parallel, delayed

from dnafiber.postprocess import refine_segmentation

from dnafiber.postprocess.fiber import FiberProps, Fibers
from dnafiber.ui.inference import ui_inference_cacheless
from dnafiber.data.utils import numpy_to_base64_png
from dnafiber.ui.utils import get_image_cacheless, get_multifile_image, _get_model
from dnafiber.postprocess.error_detection import load_model
import numpy as np


def run_one_file(
    file,
    model,
    reverse_channels=False,
    pixel_size=0.13,
    prediction_threshold=1 / 3,
    use_tta=True,
    use_correction=True,
    verbose=True,
    bit_depth=14,
):
    is_cuda_available = torch.cuda.is_available()
    if isinstance(file, np.ndarray):
        # If the file is already an image array, we don't need to load it
        image = file
        filename = "Provided Image"
    elif isinstance(file, tuple):
        if file[0] is None:
            filename = file[1].name
        if file[1] is None:
            filename = file[0].name
        image = get_multifile_image(file)
    else:
        filename = file.name
        image = get_image_cacheless(file, reverse_channels, bit_depth=bit_depth)
    start = time.time()
    prediction = ui_inference_cacheless(
        _model=model,
        _image=image,
        pixel_size=pixel_size,
        _device="cuda" if is_cuda_available else "cpu",
        use_tta=use_tta,
        use_correction=use_correction,
        only_segmentation=True,
        prediction_threshold=prediction_threshold,
        verbose=verbose,
    )

    if verbose:
        print(f"Prediction time: {time.time() - start:.2f} seconds for {filename}")
    h, w = prediction.shape
    start = time.time()
    correction_model = load_model() if use_correction else None

    y_size, x_size = 2048, 2048
    if h > y_size or w > x_size:
        # Extract blocks from the prediction

        blocks = [
            (
                image[y : y + y_size, x : x + x_size],
                prediction[y : y + y_size, x : x + x_size],
                y,
                x,
            )
            for y in range(0, h, y_size)
            for x in range(0, w, x_size)
        ]

        parallel_results = Parallel(n_jobs=4, backend="threading")(
            delayed(refine_segmentation)(
                block_img,
                block,
                x,
                y,
                correction_model=correction_model,
                verbose=verbose,
                device="cuda" if is_cuda_available else "cpu",
            )
            for (block_img, block, y, x) in (blocks)
        )

        results = Fibers(
            [fiber for block_result in parallel_results for fiber in block_result]
        )
    else:
        results = refine_segmentation(
            image,
            prediction,
            correction_model=correction_model,
            device="cuda" if is_cuda_available else "cpu",
            verbose=verbose,
        )
    if verbose:
        print(f"Refinement time: {time.time() - start:.2f} seconds for {filename}")

    return results


def format_results(results: list[FiberProps], pixel_size: float) -> pd.DataFrame:
    """
    Format the results for display in the UI.
    """
    results = [fiber for fiber in results if fiber.is_valid]
    all_results = dict(
        FirstAnalog=[], SecondAnalog=[], length=[], ratio=[], fiber_type=[]
    )
    all_results["FirstAnalog"].extend([fiber.red * pixel_size for fiber in results])
    all_results["SecondAnalog"].extend([fiber.green * pixel_size for fiber in results])
    all_results["length"].extend(
        [fiber.red * pixel_size + fiber.green * pixel_size for fiber in results]
    )
    all_results["ratio"].extend([fiber.ratio for fiber in results])
    all_results["fiber_type"].extend([fiber.fiber_type for fiber in results])

    return pd.DataFrame.from_dict(all_results)


def format_results_to_dataframe(
    _prediction,
    _image,
    resolution=400,
    include_thumbnails=True,
    pixel_size=0.13,
    include_bbox=False,
    include_segmentation=False,
):
    data = dict(
        fiber_id=[],
        firstAnalog=[],
        secondAnalog=[],
        ratio=[],
        fiber_type=[],
    )
    if include_thumbnails:
        data["Visualization"] = []
        data["Segmentation"] = []
    if include_bbox:
        data["bbox"] = []
    if include_segmentation:
        data["segmentation"] = []
    for fiber in _prediction:
        data["fiber_id"].append(fiber.fiber_id)
        r, g = fiber.counts
        red_length = pixel_size * r
        green_length = pixel_size * g
        data["firstAnalog"].append(f"{red_length:.3f} ")
        data["secondAnalog"].append(f"{green_length:.3f} ")
        data["ratio"].append(f"{green_length / red_length:.3f}")
        data["fiber_type"].append(fiber.fiber_type)
        if include_segmentation:
            data["segmentation"].append(fiber.data)
        if include_bbox:
            data["bbox"].append(fiber.bbox)

        if not include_thumbnails:
            continue

        x, y, w, h = fiber.bbox

        # Extract a region twice as large as the bbox from the image
        offsetX = math.floor(w / 2)
        offsetY = math.floor(h / 2)
        visu = _image[
            max(0, y - offsetY) : min(_image.shape[0], y + h + offsetY),
            max(0, x - offsetX) : min(_image.shape[1], x + w + offsetX),
        ]

        # Express the bbox in the same coordinate system as the visualization
        x = max(0, offsetX)
        y = max(0, offsetY)

        # Draw the bbox on the visualization
        cv2.rectangle(visu, (x, y), (x + w, y + h), (0, 0, 255), 3)
        segmentation = fiber.data
        # Scale the visualization to a minimum width of 256 pixels

        if visu.shape[1] != resolution:
            scale = resolution / visu.shape[1]
            visu = cv2.resize(
                visu,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_LINEAR,
            )
            segmentation = cv2.resize(
                segmentation,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_NEAREST_EXACT,
            )
            offsetX = math.floor(offsetX * scale)
            offsetY = math.floor(offsetY * scale)

        red_mask = segmentation == 1
        green_mask = segmentation == 2
        # Convert the segmentation to a 3-channel image
        segmentation = cv2.cvtColor(segmentation, cv2.COLOR_GRAY2BGR)
        # segmentation== 1 is red, segmentation==2 is green
        segmentation[red_mask] = np.array([255, 0, 0])
        segmentation[green_mask] = np.array([0, 255, 0])
        # Make sure the
        data["Visualization"].append(visu)
        data["Segmentation"].append(segmentation)
    df = pd.DataFrame(data)
    df = df.rename(
        columns={
            "firstAnalog": "First analog (µm)",
            "secondAnalog": "Second analog (µm)",
            "ratio": "Ratio",
            "fiber_type": "Fiber type",
            "fiber_id": "Fiber ID",
        }
    )
    if include_thumbnails:
        df["Visualization"] = df["Visualization"].apply(
            lambda x: numpy_to_base64_png(x)
        )
        df["Segmentation"] = df["Segmentation"].apply(lambda x: numpy_to_base64_png(x))
    return df


class Models(str, Enum):
    UNET_SE_RESNET101 = "unet_se_resnet101"
    UNET_SE_RESNET50 = "unet_se_resnet50"
    UNET_EFFICIENTNET_B0 = "unet_timm-efficientnet-b0"
    UNET_MOBILEONE_S0 = "unet_mobileone_s0"
    UNET_MOBILEONE_S1 = "unet_mobileone_s1"
    UNET_MOBILEONE_S2 = "unet_mobileone_s2"
    UNET_MOBILEONE_S3 = "unet_mobileone_s3"
    SEGFORMER_MIT_B0 = "segformer_mit_b0"
    SEGFORMER_MIT_B1 = "segformer_mit_b1"
    SEGFORMER_MIT_B2 = "segformer_mit_b2"
    SEGFORMER_MIT_B4 = "segformer_mit_b4"

    ENSEMBLE = "ensemble"


MODELS_ZOO = {
    "U-Net MobileOne S0": Models.UNET_MOBILEONE_S0,
    "U-Net SE-ResNet101": Models.UNET_SE_RESNET101,
    "U-Net SE-ResNet50": Models.UNET_SE_RESNET50,
    "U-Net EfficientNet B0": Models.UNET_EFFICIENTNET_B0,
    "U-Net MobileOne S1": Models.UNET_MOBILEONE_S1,
    "U-Net MobileOne S2": Models.UNET_MOBILEONE_S2,
    "U-Net MobileOne S3": Models.UNET_MOBILEONE_S3,
    "Segformer MIT B0": Models.SEGFORMER_MIT_B0,
    "Segformer MIT B1": Models.SEGFORMER_MIT_B1,
    "Segformer MIT B2": Models.SEGFORMER_MIT_B2,
    "Segformer MIT B4": Models.SEGFORMER_MIT_B4,
}
MODELS_ZOO_R = {v: k for k, v in MODELS_ZOO.items()}

ENSEMBLE = [
    # Models.UNET_MOBILEONE_S0,
    Models.UNET_SE_RESNET101,
    Models.SEGFORMER_MIT_B2,
    Models.SEGFORMER_MIT_B4,
    Models.UNET_MOBILEONE_S1,
    Models.UNET_MOBILEONE_S2,
    Models.UNET_MOBILEONE_S3,
]

from pathlib import Path
import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage.measure import label
from dnafiber.postprocess.core import extract_fibers
from tqdm.auto import tqdm
from copy import deepcopy
import pandas as pd
from dnafiber.postprocess import refine_segmentation
from dnafiber.inference import infer, transform, _get_model
from dnafiber.ui.utils import get_image_cacheless
from dnafiber.ui.inference import ui_inference_cacheless
import torch
from tqdm.contrib.concurrent import process_map, thread_map
from functools import partial

class FiberExtractor:
    def expand_bbox(self, factor=0.1):
        """Expand the bounding box by a factor."""
        for k, list_fibers in self.fibers.items():
            for fiber in list_fibers:
                bbox = fiber.fiber.bbox
                x,  y = bbox.x, bbox.y
                width, height = bbox.width, bbox.height
                new_width = int(width * (1 + factor))
                new_height = int(height * (1 + factor))
                new_x = int(x - (new_width - width) / 2)
                new_y = int(y - (new_height - height) / 2)
                bbox.x = new_x
                bbox.y = new_y
                bbox.width = new_width
                bbox.height = new_height
                fiber.fiber.bbox = bbox
       
class AIGrader(FiberExtractor):
    def __init__(self, folder, ai_name):
        folder = Path(folder)
        self.grader_name = ai_name
        self.annotations_filepaths = list(folder.rglob("*.jpeg")) + list(
            folder.rglob("*.png")
        )
        self.annotations_filepaths.sort()
        # Extract names as the path relative to the folder
        self.names = [
            str(filepath.relative_to(folder).with_suffix(""))
            for filepath in self.annotations_filepaths
        ]
        self.fibers = {}

    @torch.no_grad()
    def extract_fibers(self, model, post_process=True):
        images = []
        for name, filepath in tqdm(
            zip(self.names, self.annotations_filepaths),
            total=len(self.annotations_filepaths),
            desc="Extracting fibers from AI data",
        ):
            image = get_image_cacheless(filepath, reverse_channel=False, id=None)
            image = transform(image=image)["image"].unsqueeze(0)
            images.append(image)

        image = torch.cat(images, dim=0)
        batch_size = 16

        if image.shape[0] > batch_size:
            all_images = torch.split(image, batch_size, dim=0)
        else:
            all_images = [image]
        if not isinstance(model, list):
            model = [model]
        all_predictions = []
        for batch in tqdm(all_images, desc="Processing batches"):
            predictions = None
            batch = batch.to(
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            for m in model:
                m = _get_model(
                    device="cuda" if torch.cuda.is_available() else "cpu", revision=m
                )
                if predictions is None:
                    with torch.autocast(
                        device_type="cuda" if torch.cuda.is_available() else "cpu"
                    ):
                        with torch.inference_mode():
                            predictions = m(batch).softmax(dim=1)
                else:
                    with torch.autocast(
                        device_type="cuda" if torch.cuda.is_available() else "cpu"
                    ):
                        with torch.inference_mode():
                            predictions = m(batch).softmax(dim=1) + predictions
            predictions = (predictions / len(model)).argmax(1).cpu().numpy()
            predictions = predictions.astype(np.uint8)
            all_predictions.append(predictions)

        predictions = np.concatenate(all_predictions, axis=0).squeeze()

        predictions = [p for p in predictions]
        images = [get_image_cacheless(f, reverse_channel=False, id=None) for f in self.annotations_filepaths]

        partial_refine_segmentation = partial(refine_segmentation, post_process=post_process, threshold=5)
        fibers = thread_map(partial_refine_segmentation, images, predictions, desc="Refining fibers", max_workers=2)

        for name, fiber in zip(self.names, fibers):
            if fiber is not None:
                self.fibers[name] = fiber
            else:
                self.fibers[name] = []


class GraderData(FiberExtractor):
    def __init__(self, folder, grader_name):
        folder = Path(folder)
        self.grader_name = grader_name
        self.annotations_filepaths = list(folder.rglob("*.png"))
        self.annotations_filepaths.sort()
        # Extract names as the path relative to the folder
        self.names = [
            str(filepath.relative_to(folder).with_suffix(""))
            for filepath in self.annotations_filepaths
        ]

        self.raw_data = [
            imread_mask(filepath) for filepath in self.annotations_filepaths
        ]
        self.skeletons = [get_skeletons(data) for data in self.raw_data]
        self.fibers_map = [get_fiber(skeleton) for skeleton in self.skeletons]
        self.fibers = {}

    def extract_fibers(self):
        for name, skeleton in tqdm(
            zip(self.names, self.skeletons), desc="Extracting fibers"
        ):
            fibers = extract_fibers((skeleton>0).astype(np.uint8), skeleton, post_process=True)
            self.fibers[name] = fibers


def imread_mask(filepath):
    img = cv2.imread(str(filepath), cv2.IMREAD_UNCHANGED)[:, :, ::-1]

    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[img[:, :, 0] > 150] = 1
    mask[img[:, :, 1] > 150] = 2
    return mask


def get_skeletons(data):
    mask = skeletonize(data > 0).astype(np.uint8)
    return mask * data


def get_fiber(skeleton):
    return label(skeleton > 0, connectivity=2)


def get_bbox_IoU(fiber1, fiber2):
    bbox1 = fiber1.fiber.bbox
    bbox2 = fiber2.fiber.bbox

    x1 = max(bbox1.x, bbox2.x)
    y1 = max(bbox1.y, bbox2.y)
    x2 = min(bbox1.x + bbox1.width, bbox2.x + bbox2.width)
    y2 = min(bbox1.y + bbox1.height, bbox2.y + bbox2.height)
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = bbox1.width * bbox1.height
    area2 = bbox2.width * bbox2.height
    union_area = area1 + area2 - intersection_area
    if union_area == 0:
        return 0.0
    return intersection_area / union_area


def find_common_fibers(graders):
    # Find fibers that were found by all the graders, based on the IoU of their bounding boxes.
    # This function assumes that the graders have already extracted fibers and stored them in their `fibers` attribute.
    # Each grader's fibers are expected to be a dictionary where keys are image names and values are lists of fibers.
    dict_common_fibers = {}
    all_imgs = list(
        set(name for grader in graders for name in grader.names)
    )  # All unique image names across graders

    # Skip processing if there are no graders
    if not graders:
        return dict_common_fibers

    for name in all_imgs:
        # Skip images not present in all graders
        if not all(name in grader.fibers for grader in graders):
            continue

        # Process each fiber from first grader
        for fiber1 in graders[0].fibers[name]:
            if fiber1 is None:
                continue

            # Check if this fiber matches one in each other grader
            matches_all = True
            matching_fibers = [fiber1]

            # Check against all other graders
            for grader in graders[1:]:
                found_match = False
                for fiber2 in grader.fibers[name]:
                    if fiber2 is None:
                        continue
                    if get_bbox_IoU(fiber1, fiber2) > 0.8:
                        matching_fibers.append(fiber2)
                        found_match = True
                        break

                if not found_match:
                    matches_all = False
                    break

            # If the fiber has a match in every grader
            if matches_all:
                if name not in dict_common_fibers:
                    dict_common_fibers[name] = []
                dict_common_fibers[name].append(fiber1)

    return dict_common_fibers


def find_union_fibers(graders):
    # Find fibers that were found by at least one grader, based on the IoU of their bounding boxes.
    # This function assumes that the graders have already extracted fibers and stored them in their `fibers` attribute.
    dict_union_fibers = {}
    all_imgs = list(
        set(name for grader in graders for name in grader.names)
    )  # All unique image names across graders

    for name in all_imgs:
        # Collect all fibers from all graders for the current image
        all_fibers_for_image = []
        for grader in graders:
            if name in grader.fibers:
                all_fibers_for_image.extend(
                    [f for f in grader.fibers[name] if f is not None]
                )

        # Use IoU to find unique fibers (remove duplicates)
        unique_fibers = []
        for fiber_to_add in all_fibers_for_image:
            is_duplicate = False
            for existing_fiber in unique_fibers:
                if get_bbox_IoU(fiber_to_add, existing_fiber) > 0.75:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_fibers.append(fiber_to_add)

        if unique_fibers:
            dict_union_fibers[name] = unique_fibers

    return dict_union_fibers


def check_if_fiber_was_found_by_grader(grader, union_fibers):
    # Check if a grader has found any fibers in the union fibers.
    # This function assumes that the grader has already extracted fibers and stored them in its `fibers` attribute.
    for name, fibers in union_fibers.items():
        if name in grader.fibers:
            for fiber in fibers:
                if fiber in grader.fibers[name]:
                    return True
    return False


def format_unions_fibers(graders):
    union_fibers = find_union_fibers(graders)
    result = {f"{graders.grader_name}": [] for graders in graders}
    for name, fibers in union_fibers.items():
        for fiber in fibers:
            for grader in graders:
                if name in grader.fibers and np.any(
                    [get_bbox_IoU(fiber, f) > 0.75 for f in grader.fibers[name]]
                ):
                    result[f"{grader.grader_name}"].append(True)
                else:
                    result[f"{grader.grader_name}"].append(False)

    return pd.DataFrame(result)

def get_green_red_and_ratio_fibers(grader1, grader2):
    """
    Find common fibers across all graders and compute the green length, red length and ratio of fibers found by each grader.
    This function assumes that the graders have already extracted fibers and stored them in their `fibers` attribute.
    Each grader's fibers are expected to be a dictionary where keys are image names and values are lists of fibers.
    The function returns a DataFrame with the ratio of fibers found by each grader.
    """

    # We don't look for common fibers, as we want to know which annotator found which fibers.

    all_imgs = list(
        set(name for grader in [grader1, grader2] for name in grader.names)
    )  # All unique image names across graders
    result = {
        f"{grader1.grader_name}_green": [],
        f"{grader1.grader_name}_red": [],
        f"{grader2.grader_name}_green": [],
        f"{grader2.grader_name}_red": [],
        f"{grader1.grader_name}_ratio": [],
        f"{grader2.grader_name}_ratio": [],
    }
    for name in all_imgs:
        if name not in grader1.fibers or name not in grader2.fibers:
            continue

        fibers1 = grader1.fibers[name]
        fibers2 = grader2.fibers[name]
        for f1 in fibers1:
            for f2 in fibers2:
                if get_bbox_IoU(f1, f2) < 0.5:
                    continue
                
                result[f"{grader1.grader_name}_green"].append(f1.green)
                result[f"{grader1.grader_name}_red"].append(f1.red)
                result[f"{grader1.grader_name}_ratio"].append(f1.ratio)

                result[f"{grader2.grader_name}_ratio"].append(f2.ratio)
                result[f"{grader2.grader_name}_green"].append(f2.green)
                result[f"{grader2.grader_name}_red"].append(f2.red)


    return pd.DataFrame(result)

def assess_detection_fibers(grader: AIGrader, reference: GraderData, IoU_threshold=0.5):
    names = list(set(grader.names) | set(reference.names))
    IoUs = []
    precision = []
    recall = []
    for name in tqdm(names, desc="Assessing detection fibers"):
        if name not in grader.fibers or name not in reference.fibers:
            continue
        
        TP = 0
        P = 0
        FP = 0
        IoU = []
        fibers1 = grader.fibers[name]
        fibers2 = reference.fibers[name]
        PredP = len(fibers1)
        P = len(fibers2)
        already_found_f1 =  np.zeros(len(fibers1), dtype=bool)
        already_found_f2= np.zeros(len(fibers2), dtype=bool)
        for i, f1 in enumerate(fibers1):
            for j, f2 in enumerate(fibers2):
                if already_found_f2[j] or already_found_f1[i]:
                    continue
                iou = get_bbox_IoU(f1, f2)
                
                if iou > IoU_threshold:
                    already_found_f1[i] = True
                    already_found_f2[j] = True
                    IoU.append(iou)
                    TP += 1
                else:
                    FP += 1
        
        precision.append(TP / PredP if PredP > 0 else 0)
        recall.append(TP / P if P > 0 else 0)
        IoUs.append(np.mean(IoU) if IoU else 0)
                
    return {"precision": np.array(precision).squeeze(),
            "recall": np.array(recall).squeeze(),
            "IoU":  np.array(IoUs).squeeze()}
                    


from tqdm.auto import tqdm
from dnafiber.data.utils import read_colormask
import numpy as np


def build_consensus_map(intergraders, root_img, list_img):
    all_masks = []
    for img_path in tqdm(list_img):
        path_from_root = img_path.relative_to(root_img)
        masks = []
        for intergrader in intergraders:
            intergrader_path = (intergrader / path_from_root).with_suffix(".png")
            if not intergrader_path.exists():
                print(f"Missing {intergrader_path}")
                continue
            mask = read_colormask(intergrader_path)
            masks.append(mask)
        masks = np.array(masks)

        all_masks.append(masks)
    return np.array(all_masks)

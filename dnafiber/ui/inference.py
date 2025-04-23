from dnafiber.ui.utils import get_model
from monai.inferers import SlidingWindowInferer
import albumentations as A
import torch
import streamlit as st
import torch.nn.functional as F
from skimage.measure import label
from skimage.segmentation import expand_labels
from skimage.morphology import skeletonize
import cv2

transform = A.Compose(
    [
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.ToTensorV2(),
    ]
)


@st.cache_data
def inference(model, _image, _device, id=None):
    model = get_model(_device, model)
    model_pixel_size = 0.26
    scale = st.session_state.get("pixel_size", 0.13) / model_pixel_size
    tensor = transform(image=_image)["image"].unsqueeze(0).to(_device)
    h, w = tensor.shape[2], tensor.shape[3]
    tensor = F.interpolate(
        tensor,
        size=(int(h * scale), int(w * scale)),
        mode="bilinear",
    )
    with st.spinner("Sliding window segmentation in progress..."):
        with torch.no_grad():
            inferer = SlidingWindowInferer(
                roi_size=(512, 512),
                sw_batch_size=1,
                overlap=0.5,
                mode="gaussian",
                device=_device,
                progress=True,
            )
            output = inferer(tensor, model)
            output = torch.argmax(output, dim=1).squeeze().cpu().numpy()
    with st.spinner("Post-processing in progress..."):
        output = post_process(output)
    output = cv2.resize(
        output,
        (w, h),
        interpolation=cv2.INTER_NEAREST_EXACT,
    )
    return output


def post_process(prediction_map):
    prediction_map = expand_labels(prediction_map, distance=5)
    binary_mask = prediction_map > 0
    skeleton = skeletonize(binary_mask)
    prediction_map = skeleton * prediction_map
    labeled_mask, count = label(skeleton, return_num=True, connectivity=2)
    for i in range(1, count + 1):
        current_fiber = prediction_map[labeled_mask == i]
        # Check if the fiber is too small or too large
        if current_fiber.size < 10:
            prediction_map[labeled_mask == i] = 0
            continue
        _, n_segments = label(current_fiber, return_num=True, connectivity=1)
        if n_segments == 1 or n_segments > 3:
            prediction_map[labeled_mask == i] = 0
            continue

    return prediction_map

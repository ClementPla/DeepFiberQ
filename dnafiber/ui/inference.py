import streamlit as st
from dnafiber.inference import infer
from dnafiber.postprocess.core import refine_segmentation
import numpy as np


def ui_inference(_model, _image, _device, use_post_processing, id=None):
    h, w = _image.shape[:2]
    with st.spinner("Sliding window segmentation in progress..."):
        output = infer(
            _model,
            image=_image,
            device=_device,
            scale=st.session_state.get("pixel_size", 0.13),
        )
    output = output.astype(np.uint8)

    output = refine_segmentation(output, fix_junctions=use_post_processing)
    return output

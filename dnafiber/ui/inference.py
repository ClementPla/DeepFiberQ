import streamlit as st
from dnafiber.inference import infer
from dnafiber.postprocess.core import refine_segmentation
import numpy as np
from dnafiber.ui.utils import _get_model
import torch
from dnafiber.postprocess.error_detection import load_model

@st.cache_data
def ui_inference(_model, _image, _device, use_tta=True, use_correction=True, id=None):
    return ui_inference_cacheless(_model, _image, _device, use_tta=use_tta, use_correction=use_correction)


@st.cache_resource
def get_model(model_name):
    model = _get_model(
        device="cuda" if torch.cuda.is_available() else "cpu",
        revision=model_name,
    )
    return model


def ui_inference_cacheless(
    _model, _image, _device, use_tta=True, only_segmentation=False, use_correction=None
):
    """
    A cacheless version of the ui_inference function.
    This function does not use caching and is intended for use in scenarios where caching is not desired.
    """
    if use_correction:
        correction_model = load_model()
    else:
        correction_model = None
    h, w = _image.shape[:2]
    with st.spinner("Sliding window segmentation in progress..."):
        if isinstance(_model, list):
            output = None
            for model in _model:
                with st.spinner(text="Segmenting with model: {}".format(model)):
                    if isinstance(model, str):
                        model = get_model(model)
                    if output is None:
                        output = infer(
                            model,
                            image=_image,
                            device=_device,
                            use_tta=use_tta,
                            scale=st.session_state.get("pixel_size", 0.13),
                            only_probabilities=True,
                        ).cpu()
                    else:
                        output = (
                            output
                            + infer(
                                model,
                                image=_image,
                                device=_device,
                                use_tta=use_tta,
                                scale=st.session_state.get("pixel_size", 0.13),
                                only_probabilities=True,
                            ).cpu()
                        )
            output = (output / len(_model)).argmax(1).squeeze().numpy()
        else:
            output = infer(
                _model,
                image=_image,
                device=_device,
                scale=st.session_state.get("pixel_size", 0.13),
            )
    output = output.astype(np.uint8)
    if only_segmentation:
        return output
    with st.spinner("Post-processing segmentation..."):
        output = refine_segmentation(_image, output, correction_model=correction_model, device=_device)
    return output

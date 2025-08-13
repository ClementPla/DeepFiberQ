import streamlit as st
from dnafiber.inference import infer
from dnafiber.postprocess.core import refine_segmentation
import numpy as np
from dnafiber.deployment import _get_model
import torch


@st.cache_data
def ui_inference(_model, _image, _device, use_tta=True, threshold=10, id=None):
    return ui_inference_cacheless(_model, _image, _device, threshold=threshold)


@st.cache_resource
def get_model(model_name):
    model = _get_model(
        device="cuda" if torch.cuda.is_available() else "cpu",
        revision=model_name,
    )
    return model


def ui_inference_cacheless(
    _model, _image, _device, use_tta=True, threshold=10, only_segmentation=False
):
    """
    A cacheless version of the ui_inference function.
    This function does not use caching and is intended for use in scenarios where caching is not desired.
    """
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
        output = refine_segmentation(_image, output, threshold=threshold)
    return output

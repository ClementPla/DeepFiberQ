import streamlit as st
from dnafiber.inference import infer
from dnafiber.postprocess.core import refine_segmentation
import numpy as np
from dnafiber.ui.utils import _get_model
import torch
from dnafiber.postprocess.error_detection import load_model
from skimage.segmentation import expand_labels
from skimage.morphology import binary_closing, remove_small_objects


@st.cache_resource
def ui_inference(
    _model,
    _image,
    _device,
    use_tta=True,
    use_correction=True,
    prediction_threshold=1 / 3,
    id=None,
):
    return ui_inference_cacheless(
        _model,
        _image,
        _device,
        pixel_size=st.session_state.get("pixel_size", 0.13),
        use_tta=use_tta,
        prediction_threshold=prediction_threshold,
        use_correction=use_correction,
    )


@st.cache_resource
def get_model(model_name):
    model = _get_model(
        device="cuda" if torch.cuda.is_available() else "cpu",
        revision=model_name,
    )
    return model


def ui_inference_cacheless(
    _model,
    _image,
    _device,
    pixel_size,
    use_tta=True,
    only_segmentation=False,
    use_correction=None,
    prediction_threshold=1 / 3,
    verbose=True,
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
                        model.eval()
                    if output is None:
                        output = infer(
                            model,
                            image=_image,
                            device=_device,
                            use_tta=use_tta,
                            scale=pixel_size,
                            only_probabilities=True,
                            verbose=verbose,
                        ).cpu()
                    else:
                        output = (
                            output
                            + infer(
                                model,
                                image=_image,
                                device=_device,
                                use_tta=use_tta,
                                scale=pixel_size,
                                only_probabilities=True,
                                verbose=verbose,
                            ).cpu()
                        )
            output = output / len(_model)

        else:
            output = infer(
                _model,
                image=_image,
                device=_device,
                scale=pixel_size,
                use_tta=use_tta,
                only_probabilities=True,
                verbose=verbose,
            )

        output = output.cpu().numpy()

        pos_prob = 1 - output[0, 0, :, :]

        pos_prob = binary_closing(pos_prob >= prediction_threshold, np.ones((3, 3)))
        pos_prob = remove_small_objects(pos_prob, min_size=50)

        output[0, 0, pos_prob > 0] = 0
        output[0, 0, pos_prob == 0] = 1

        output = np.argmax(output, axis=1).squeeze()
    output = output.astype(np.uint8)
    # output = expand_labels(output, distance=5).astype(np.uint8)
    if only_segmentation:
        return output
    with st.spinner("Post-processing segmentation..."):
        output = refine_segmentation(
            _image, output, correction_model=correction_model, device=_device
        )
    return output

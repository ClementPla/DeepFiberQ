import streamlit as st
from dnafiber.inference import infer
from dnafiber.postprocess.core import refine_segmentation
import numpy as np
import torch


@st.cache_data
def ui_inference(_model, _image, _device, postprocess=True, id=None):
    h, w = _image.shape[:2]
    with st.spinner("Sliding window segmentation in progress..."):
        if isinstance(_model, list):
            output = None
            for model in _model:
                with st.spinner(text="Segmenting with model: {}".format(model)):
                    if output is None:
                        output = infer(
                            model,
                            image=_image,
                            device=_device,
                            scale=st.session_state.get("pixel_size", 0.13),
                            only_probabilities=True,
                        ).cpu()
                    else:
                        output = output + infer(
                            model,
                            image=_image,
                            device=_device,
                            scale=st.session_state.get("pixel_size", 0.13),
                            only_probabilities=True,
                        ).cpu()
                        
                    
            output = (output / len(model)).argmax(1).squeeze().numpy()
        else:
            output = infer(
                _model,
                image=_image,
                device=_device,
                scale=st.session_state.get("pixel_size", 0.13),
            )
    output = output.astype(np.uint8)
    if postprocess:
        with st.spinner("Post-processing segmentation..."):
            output = refine_segmentation(output, fix_junctions=postprocess)
    return output

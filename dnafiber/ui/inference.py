import streamlit as st
from dnafiber.inference import post_process, infer
from skimage.morphology import skeletonize


@st.cache_data
def ui_inference(model, _image, _device, use_post_processing, id=None):
    h, w = _image.shape[:2]
    with st.spinner("Sliding window segmentation in progress..."):
        output = infer(
            model,
            image=_image,
            device=_device,
            scale=st.session_state.get("pixel_size", 0.13),
        )
    if use_post_processing:
        with st.spinner("Post-processing in progress..."):
            output = post_process(output)
    else:
        output = skeletonize(output > 0, method="lee") * output

    return output

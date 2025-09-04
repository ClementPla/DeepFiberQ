import math

import cv2
import numpy as np
import streamlit as st
import streamlit_image_coordinates
import torch
from bokeh.layouts import gridplot, column, row
from bokeh.models import (
    HoverTool,
    Range1d,
)
from bokeh.plotting import figure
from catppuccin import PALETTE
from skimage.segmentation import expand_labels
from skimage.util import view_as_blocks
from streamlit_bokeh import streamlit_bokeh
from PIL import Image
import io
from dnafiber.data.utils import CMAP, pad_image_to_croppable
from dnafiber.deployment import MODELS_ZOO, MODELS_ZOO_R, ENSEMBLE
from dnafiber.ui.components import show_fibers_cacheless, table_components
from dnafiber.ui.inference import get_model, ui_inference, ui_inference_cacheless
from dnafiber.ui.utils import (
    get_image,
    get_multifile_image,
    get_resized_image,
)
from dnafiber.ui.custom import fiber_ui

st.set_page_config(
    layout="wide",
    page_icon=":microscope:",
)

st.markdown(
    """
        <style>
               .block-container {
                    padding-top: 1rem;
                    padding-bottom: 0rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
        </style>
        """,
    unsafe_allow_html=True,
)


def on_session_start():
    can_start = (
        st.session_state.get("files_uploaded", None) is not None
        and len(st.session_state.files_uploaded) > 0
    )

    if can_start:
        return can_start

    cldu_exists = (
        st.session_state.get("files_uploaded_cldu", None) is not None
        and len(st.session_state.files_uploaded_cldu) > 0
    )
    idu_exists = (
        st.session_state.get("files_uploaded_idu", None) is not None
        and len(st.session_state.files_uploaded_idu) > 0
    )

    if cldu_exists and idu_exists:
        if len(st.session_state.get("files_uploaded_cldu")) != len(
            st.session_state.get("files_uploaded_idu")
        ):
            st.error("Please upload the same number of CldU and IdU files.")
            return False


def start_inference(use_tta=True, use_correction=True, prediction_threshold=1 / 3):
    image = st.session_state.image_inference
    org_h, org_w = image.shape[:2]
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    if "ensemble" in st.session_state.model_name:
        model = []
        for _ in range(len(ENSEMBLE)):
            with st.spinner(f"Loading model {_ + 1}/{len(ENSEMBLE)}..."):
                model.append(get_model(ENSEMBLE[_]))
    else:
        with st.spinner("Loading model..."):
            model = get_model(st.session_state.model_name)
    prediction = ui_inference(
        model,
        image,
        "cuda" if torch.cuda.is_available() else "cpu",
        use_tta=use_tta,
        use_correction=use_correction,
        prediction_threshold=prediction_threshold,
        id=st.session_state.image_id,
    )

    tab_viewer, tab_fibers = st.tabs(["Viewer", "Fibers"])
    with tab_fibers:
        df = show_fibers_cacheless(
            _prediction=prediction,
            _image=image,
            image_id=st.session_state.image_id,
        )
        table_components(df)

    with tab_viewer:
        if image.max() > 25:
            with st.spinner("Normalizing image..."):
                image = cv2.normalize(
                    image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
                )

        max_size = 20000
        h, w = image.shape[:2]
        size = max(h, w)
        scale = 1.0
        if size > max_size:
            st.toast(
                f"Images are displayed at a lower resolution of {max_size} pixel wide"
            )
            scale = max_size / size
            with st.spinner("Resizing image..."):
                image = cv2.resize(
                    image,
                    None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_LINEAR,
                )

        fiber_ui(image, prediction.valid_copy().svgs(scale=scale))
        with st.spinner("Preparing downloadable results..."):
            labelmap = prediction.filtered_copy().get_labelmap(org_h, org_w, 3)
            labelmap = (CMAP(labelmap)[:, :, :3] * 255).astype(np.uint8)

            image = Image.fromarray(labelmap)

            # Create an in-memory binary stream
            buffer = io.BytesIO()

            # Save the image to the buffer in PNG format
            image.save(buffer, format="jpeg")

            # Get the byte data from the buffer
            jpeg_data = buffer.getvalue()
        with st.expander("Download results", expanded=True):
            st.download_button(
                "Download segmented image",
                data=jpeg_data,
                file_name="segmented_image.jpg",
                mime="image/jpeg",
            )


def create_display_files(files):
    if files is None or len(files) == 0:
        return "No files uploaded"
    display_files = []
    for file in files:
        if isinstance(file, tuple):
            if file[0] is None:
                name = f"Second analog only {file[1].name}"
            elif file[1] is None:
                name = f"First analog only {file[0].name}"
            else:
                name = f"{file[0].name} and {file[1].name}"
            display_files.append(name)
        else:
            display_files.append(file.name)
    return display_files


if on_session_start():
    files = st.session_state.files_uploaded
    displayed_names = create_display_files(files)
    with st.sidebar:
        selected_file = st.selectbox(
            "Pick an image",
            displayed_names,
            index=0,
            help="Select an image to view and analyze.",
        )

    # Find index of the selected file
    index = displayed_names.index(selected_file)
    file = files[index]
    if isinstance(file, tuple):
        file_id = str(hash(file[0]))
        if file[0] is None or file[1] is None:
            missing = "First analog" if file[0] is None else "Second analog"
            st.warning(
                f"In this image, {missing} channel is missing. We assume the intended goal is to segment the DNA fibers without differentiation. \
                       Note the model may still predict two classes and try to compute a ratio; these informations can be ignored."
            )
        image = get_multifile_image(file)
    else:
        file_id = str(hash(file))
        image = get_image(
            file,
            reverse_channel=st.session_state.get("reverse_channels", False),
            id=file_id,
            bit_depth=st.session_state.get("bit_depth", 14),
        )

    org_image = image.copy()
    h, w = image.shape[:2]
    with st.sidebar:
        st.metric(
            "Pixel size (µm)",
            st.session_state.get("pixel_size", 0.13),
        )
        st.metric(
            "Bit depth",
            st.session_state.get("bit_depth", 14),
        )

    thumbnail = get_resized_image(image, file_id)

    with st.sidebar:
        with st.expander("Model", expanded=True):
            use_ensemble = st.checkbox(
                "Ensemble model",
                value=False,
                help="Use all available models to improve segmentation results.",
            )
            model_name = st.selectbox(
                "Select a model",
                list(MODELS_ZOO.values()),
                format_func=lambda x: MODELS_ZOO_R[x],
                index=0,
                help="Select a model to use for inference",
                disabled=use_ensemble,
            )
            if use_ensemble:
                st.warning(
                    "Ensemble model is selected. All available models will be used for inference."
                )
                model_name = "ensemble"

            use_tta = st.checkbox(
                "Use test time augmentation (TTA)",
                value=False,
                help="Use test time augmentation to improve segmentation results.",
            )

            detect_errors = st.checkbox(
                "Use error detection and filtering",
                value=True,
                help="Use the error filtering model to improve detection results.",
            )

            prediction_threshold = st.slider(
                "Prediction threshold",
                min_value=0.0,
                max_value=1.0,
                value=1 / 3,
                step=0.01,
                help="Select the prediction threshold for the model. Lower values may increase the number of detected fibers.",
            )

            col1, col2 = st.columns(2)
            with col1:
                st.write("Running on:")
            with col2:
                st.button(
                    "GPU" if torch.cuda.is_available() else "CPU",
                    disabled=True,
                )

    # image = blocks[which_y, which_x, 0]
    with st.sidebar:
        st.image(image, caption="Current image", use_container_width=True)

    st.session_state.model_name = model_name
    st.session_state.image_inference = image
    st.session_state.image_id = (
        (file_id + str(model_name))
        + ("_use_tta" if use_tta else "_no_tta")
        + ("_detect_errors" if detect_errors else "_no_detect_errors")
        + f"_{prediction_threshold:.2f}"
        + f"_{h}x{w}"
        + f"_{st.session_state.get('pixel_size', 0.13)}um"
        + f"_{'ensemble' if use_ensemble else model_name}"
        + f"_{'rev' if st.session_state.get('reverse_channels', False) else 'no_rev'}"
        + f"_{st.session_state.get('bit_depth', 14)}bit"
    )
    col1, col2, col3 = st.columns([1, 1, 1])

    start_inference(
        use_tta=use_tta,
        use_correction=detect_errors,
        prediction_threshold=prediction_threshold,
    )


else:
    st.switch_page("pages/1_Load.py")

# Add a callback to mouse move event

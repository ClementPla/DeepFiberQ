import cv2
import numpy as np
import streamlit as st
import torch
from dnafiber.data.utils import CMAP
from PIL import Image
import io

from dnafiber.deployment import MODELS_ZOO, MODELS_ZOO_R, ENSEMBLE, Models
from dnafiber.ui.components import (
    show_fibers,
    table_components,
    distribution_analysis,
    viewer_components,
)
from dnafiber.ui.custom.fiber_ui import fiber_ui
from dnafiber.ui.inference import get_model, ui_inference
from dnafiber.ui.utils import (
    build_file_id,
    build_inference_id,
    get_image,
    get_multifile_image,
    get_resized_image,
)
from dnafiber.ui import DefaultValues as DV
from dnafiber.ui.utils import retain_session_state

retain_session_state(st.session_state)
st.set_page_config(
    layout="wide",
    page_icon=":microscope:",
)

st.markdown(
    """
        <style>
               .block-container {
                    padding-top: 3rem;
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
        st.session_state.get("analog_2_files", None) is not None
        and len(st.session_state.analog_2_files) > 0
    )
    idu_exists = (
        st.session_state.get("analog_1_files", None) is not None
        and len(st.session_state.analog_1_files) > 0
    )

    if cldu_exists and idu_exists:
        if len(st.session_state.get("analog_2_files")) != len(
            st.session_state.get("analog_1_files")
        ):
            st.error("Please upload the same number of CldU and IdU files.")
            return False


def clear_cache():
    st.cache_data.clear()
    st.cache_resource.clear()
    st.success("Cache cleared!")


def start_inference(
    image,
    model_name,
    use_tta=DV.USE_TTA,
    use_correction=DV.USE_CORRECTION,
    prediction_threshold=DV.PREDICTION_THRESHOLD,
    inference_id=None,
):
    org_h, org_w = image.shape[:2]
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    if model_name == Models.ENSEMBLE:
        model = []
        for _ in range(len(ENSEMBLE)):
            with st.spinner(f"Loading model {_ + 1}/{len(ENSEMBLE)}..."):
                model.append(get_model(ENSEMBLE[_]))
    else:
        with st.spinner("Loading model..."):
            model = get_model(model_name)

    prediction = ui_inference(
        model,
        image,
        "cuda" if torch.cuda.is_available() else "cpu",
        use_tta,
        use_correction,
        prediction_threshold,
        key=inference_id,
    )

    tab_viewer, tab_fibers, tab_distributions = st.tabs(
        ["Viewer", "Fibers", "Distribution"]
    )

    with tab_viewer:
        max_dim = max(org_h, org_w)
        max_size = 10000
        if max_dim > max_size:
            st.toast(
                f"Images are displayed at a lower resolution of {max_size} pixel wide"
            )
        rescaled_image, scale = viewer_components(image, prediction, inference_id)
        selected_fibers = fiber_ui(
            rescaled_image, prediction.valid_copy().svgs(scale=scale), key=inference_id
        )
    for fiber in prediction:
        if fiber.fiber_id in selected_fibers:
            fiber.is_an_error = True

    with tab_fibers:
        df = show_fibers(
            _prediction=prediction,
            _image=image,
            inference_id=inference_id,
        )
        for idx in df.index:
            if df.at[idx, "Fiber ID"] in selected_fibers:
                df.at[idx, "is_valid"] = False
        table_components(df)

        st.write("Fibers removed from the viewer are marked as errors.")
        st.data_editor(selected_fibers, hide_index=True)

    with tab_distributions:
        distribution_analysis(prediction)

    with st.sidebar:
        with st.expander("Download results", expanded=False):
            width = st.slider(
                "Set fiber width for visualization",
                min_value=1,
                max_value=50,
                value=3,
                step=1,
            )

            prepare_download = st.button(
                "Prepare download", help="Prepare all results for download."
            )
            if prepare_download:
                with st.spinner("Preparing files..."):
                    import time

                    start = time.time()
                    labelmap = prediction.filtered_copy().get_labelmap(
                        org_h, org_w, width
                    )
                    labelmap = CMAP(labelmap, bytes=True)[:, :, :3]
                    labelmap = Image.fromarray(labelmap)

                    # Create an in-memory binary stream
                    buffer = io.BytesIO()

                    # Save the image to the buffer in PNG format
                    labelmap.save(buffer, format="jpeg")

                    # Get the byte data from the buffer
                    labelmap_jpeg_data = buffer.getvalue()

                    st.success(
                        f"Segmentation map is ready! ({time.time() - start:.2f}s)"
                    )
                    start = time.time()
                    bbox_map = prediction.filtered_copy().get_bounding_boxes_map(
                        org_h, org_w, width=width, image=image
                    )
                    bbox_map = Image.fromarray(bbox_map)
                    buffer = io.BytesIO()
                    bbox_map.save(buffer, format="jpeg")
                    bbox_jpeg_data = buffer.getvalue()
                    st.success(
                        f"Bounding boxes map is ready! ({time.time() - start:.2f}s)"
                    )
                st.download_button(
                    "Segmentation map",
                    data=labelmap_jpeg_data,
                    file_name="segmentation_map.jpg",
                    mime="image/jpeg",
                )

                st.download_button(
                    "Bounding boxes map",
                    data=bbox_jpeg_data,
                    file_name="bounding_boxes_map.jpg",
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

    file_id = build_file_id(
        file,
        pixel_size=st.session_state.get("pixel_size", DV.PIXEL_SIZE),
        reverse_channels=st.session_state.get("reverse_channels", DV.REVERSE_CHANNELS),
        bit_depth=st.session_state.get("bit_depth", DV.BIT_DEPTH),
    )
    if isinstance(file, tuple):
        if file[0] is None or file[1] is None:
            missing = "First analog" if file[0] is None else "Second analog"
            st.warning(
                f"In this image, {missing} channel is missing. We assume the intended goal is to segment the DNA fibers without differentiation. \
                       Note the model may still predict two classes and try to compute a ratio; these informations can be ignored."
            )
        image = get_multifile_image(file)
    else:
        image = get_image(
            file,
            reverse_channel=st.session_state.get(
                "reverse_channels", DV.REVERSE_CHANNELS
            ),
            bit_depth=st.session_state.get("bit_depth", DV.BIT_DEPTH),
            id=file_id,
        )

    with st.sidebar:
        st.metric(
            "Pixel size (µm)",
            st.session_state.get("pixel_size", DV.PIXEL_SIZE),
        )
        st.metric(
            "Bit depth",
            st.session_state.get("bit_depth", DV.BIT_DEPTH),
        )

    thumbnail = get_resized_image(image, file_id)

    with st.sidebar:
        with st.expander("Model", expanded=True):
            st.checkbox(
                "Ensemble model",
                key="use_ensemble",
                help="Use all available models to improve segmentation results.",
            )
            model_name = st.selectbox(
                "Select a model",
                list(MODELS_ZOO.values()),
                format_func=lambda x: MODELS_ZOO_R[x],
                index=0,
                help="Select a model to use for inference",
                disabled=st.session_state.get("use_ensemble", DV.USE_ENSEMBLE),
            )
            if st.session_state.get("use_ensemble", DV.USE_ENSEMBLE):
                st.warning(
                    "Ensemble model is selected. All available models will be used for inference."
                )
                model_name = Models.ENSEMBLE

            st.checkbox(
                "Use test time augmentation (TTA)",
                key="use_tta",
                help="Use test time augmentation to improve segmentation results.",
            )

            st.checkbox(
                "Use error detection and filtering",
                key="use_correction",
                help="Use the error filtering model to improve detection results.",
            )

            st.slider(
                "Prediction threshold",
                min_value=0.0,
                max_value=1.0,
                key="prediction_threshold",
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

    inference_id = build_inference_id(
        file_id,
        str(model_name),
        st.session_state.get("use_tta", DV.USE_TTA),
        st.session_state.get("use_correction", DV.USE_CORRECTION),
        st.session_state.get("prediction_threshold", DV.PREDICTION_THRESHOLD),
    )
    col1, col2, col3 = st.columns([1, 1, 1])

    start_inference(
        image=image,
        model_name=model_name,
        use_tta=st.session_state.get("use_tta", DV.USE_TTA),
        use_correction=st.session_state.get("use_correction", DV.USE_CORRECTION),
        prediction_threshold=st.session_state.get(
            "prediction_threshold", DV.PREDICTION_THRESHOLD
        ),
        inference_id=inference_id,
    )


else:
    st.switch_page("pages/1_Load.py")

# Add a callback to mouse move event

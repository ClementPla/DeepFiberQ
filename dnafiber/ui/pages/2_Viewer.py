import math

import cv2
import numpy as np
import streamlit as st
import streamlit_image_coordinates
import torch
from bokeh.layouts import gridplot
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
from dnafiber.deployment import MODELS_ZOO, MODELS_ZOO_R
from dnafiber.ui.components import show_fibers_cacheless, table_components
from dnafiber.ui.inference import get_model, ui_inference, ui_inference_cacheless
from dnafiber.ui.utils import (
    bokeh_imshow,
    get_image,
    get_multifile_image,
    get_resized_image,
)

st.set_page_config(
    layout="wide",
    page_icon=":microscope:",
)
st.title("Viewer")


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
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    if "ensemble" in st.session_state.model_name:
        model = []
        for _ in range(len(MODELS_ZOO)):
            with st.spinner(f"Loading model {_ + 1}/{len(MODELS_ZOO)}..."):
                model.append(get_model(MODELS_ZOO[list(MODELS_ZOO.keys())[_]]))
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

    prediction = [
        p
        for p in prediction
        if (p.fiber_type != "single") and p.fiber_type != "multiple"
    ]
    show_errors = st.checkbox(
        "Show fibers with errors",
        value=True,
        help="Show fibers that were filtered out by the error detection model.",
    )
    tab_viewer, tab_fibers = st.tabs(["Viewer", "Fibers"])
    with tab_fibers:
        df = show_fibers_cacheless(
            _prediction=prediction,
            _image=image,
            image_id=st.session_state.image_id,
            show_errors=show_errors,
        )
        table_components(df)

    with tab_viewer:
        max_width = 10000
        if image.shape[1] > max_width:
            st.toast("Images are displayed at a lower resolution of 2048 pixel wide")

        fig = display_prediction(
            _prediction=prediction,
            _image=image,
            image_id=st.session_state.image_id,
            show_errors=show_errors,
        )
        streamlit_bokeh(fig, use_container_width=True)


def display_prediction(_prediction, _image, image_id=None, show_errors=True):
    max_width = 2048
    image = _image
    if image.max() > 25:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    scale = 1
    # Resize the image to max_width
    if image.shape[1] > max_width:
        scale = max_width / image.shape[1]
        image = cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_LINEAR,
        )

    h, w = image.shape[:2]
    labels_maps = np.zeros((h, w), dtype=np.uint8)
    for i, region in enumerate(_prediction):
        if not show_errors and not region.is_valid:
            continue
        x, y, w, h = region.scaled_coordinates(scale)
        data = cv2.resize(
            expand_labels(region.data, 2),
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_NEAREST,
        )
        labels_maps[
            y : y + data.shape[0],
            x : x + data.shape[1],
        ][data > 0] = data[data > 0]
    p1 = figure(
        x_range=Range1d(-image.shape[1] / 8, image.shape[1] * 1.125, bounds="auto"),
        y_range=Range1d(image.shape[0] * 1.125, -image.shape[0] / 8, bounds="auto"),
        title=f"Detected fibers: {len(_prediction)} ({sum([1 for r in _prediction if not r.is_valid])} filtered)",
        tools="pan,wheel_zoom,box_zoom,reset,save",
        active_scroll="wheel_zoom",
    )

    p1.image(
        image=[labels_maps],
        x=0,
        y=0,
        dw=labels_maps.shape[1],
        dh=labels_maps.shape[0],
        palette=["black", st.session_state["color1"], st.session_state["color2"]]
        if np.max(labels_maps) > 0
        else ["black"],
    )
    p2 = figure(
        x_range=p1.x_range,
        y_range=p1.y_range,
        tools="pan,wheel_zoom,box_zoom,reset,save",
        active_scroll="wheel_zoom",
    )
    bokeh_imshow(p2, image)
    colors = [c.hex for c in PALETTE.latte.colors][:14]
    data_source = dict(
        x=[],
        y=[],
        width=[],
        height=[],
        color=[],
        firstAnalog=[],
        secondAnalog=[],
        ratio=[],
        fill_color=[],
        fiber_id=[],
        line_width=[],
    )
    np.random.shuffle(colors)
    for i, region in enumerate(_prediction):
        if not show_errors and not region.is_valid:
            continue
        color = colors[i % len(colors)]
        x, y, w, h = region.scaled_coordinates(scale)
        is_valid = region.is_valid
        fiberId = region.fiber_id
        data_source["x"].append((x + w / 2))
        data_source["y"].append((y + h / 2))
        data_source["width"].append(w)
        data_source["height"].append(h)
        data_source["color"].append(color if is_valid else "#ffffff")
        data_source["line_width"].append(2 if is_valid else 3)
        data_source["fill_color"].append("#ffffff00" if is_valid else ("#ff0000b2"))
        r, g = region.counts
        red_length = st.session_state["pixel_size"] * r / scale
        green_length = st.session_state["pixel_size"] * g / scale
        data_source["firstAnalog"].append(f"{red_length:.2f} µm")
        data_source["secondAnalog"].append(f"{green_length:.2f} µm")
        data_source["ratio"].append(f"{green_length / red_length:.2f}")
        data_source["fiber_id"].append(fiberId)

    rect1 = p1.rect(
        x="x",
        y="y",
        width="width",
        height="height",
        source=data_source,
        fill_color="fill_color",
        line_color="color",
        line_width="line_width",
    )
    rect2 = p2.rect(
        x="x",
        y="y",
        width="width",
        height="height",
        source=data_source,
        fill_color=None,
        line_color="color",
        line_width="line_width",
    )

    hover = HoverTool(
        tooltips=f'<b>Fiber ID: @fiber_id</b><br><p style="color:{st.session_state["color1"]};">@firstAnalog</p> <p style="color:{st.session_state["color2"]};">@secondAnalog</p><b> Ratio: @ratio</b>',
    )
    hover.renderers = [rect1, rect2]
    hover.point_policy = "follow_mouse"
    hover.attachment = "vertical"
    p1.add_tools(hover)
    p2.add_tools(hover)

    p1.axis.visible = False
    p2.axis.visible = False
    fig = gridplot(
        [[p2, p1]],
        merge_tools=True,
        sizing_mode="stretch_width",
        toolbar_options=dict(logo=None, help=None),
    )
    return fig


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
        )

    download_button = st.button(
        "Download full segmentation",
        help="Download the full segmented image as a PNG file.",
    )
    org_image = image.copy()
    h, w = image.shape[:2]
    with st.sidebar:
        st.metric(
            "Pixel size (µm)",
            st.session_state.get("pixel_size", 0.13),
        )

        block_size = st.slider(
            "Block size",
            min_value=256,
            max_value=min(8192, max(h, w)),
            value=min(2048, max(h, w)),
            step=256,
        )
    if h < block_size:
        block_size = h
    if w < block_size:
        block_size = w

    bx = by = block_size
    image = pad_image_to_croppable(image, bx, by, uid=file_id + str(bx) + str(by))
    thumbnail = get_resized_image(image, file_id)

    blocks = view_as_blocks(image, (bx, by, 3))
    x_blocks, y_blocks = blocks.shape[0], blocks.shape[1]
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

        which_y = st.session_state.get("which_y", 0)
        which_x = st.session_state.get("which_x", 0)

        # Display the selected block
        # Scale factor
        h, w = image.shape[:2]
        small_h, small_w = thumbnail.shape[:2]
        scale_h = h / small_h
        scale_w = w / small_w
        # Calculate the coordinates of the block
        y1 = math.floor(which_y * bx / scale_h)
        y2 = math.floor((which_y + 1) * bx / scale_h)
        x1 = math.floor(which_x * by / scale_w)
        x2 = math.floor((which_x + 1) * by / scale_w)
        # Draw a rectangle around the selected block

        # Check if the coordinates are within the bounds of the image
        while y2 > small_h:
            which_y -= 1
            y1 = math.floor(which_y * bx / scale_h)
            y2 = math.floor((which_y + 1) * bx / scale_h)
        while x2 > small_w:
            which_x -= 1
            x1 = math.floor(which_x * by / scale_w)
            x2 = math.floor((which_x + 1) * by / scale_w)

        st.session_state["which_x"] = which_x
        st.session_state["which_y"] = which_y

        # Draw a grid on the thumbnail
        for i in range(0, small_h, int(bx // scale_h)):
            cv2.line(thumbnail, (0, i), (small_w, i), (255, 255, 255), 1)
        for i in range(0, small_w, int(by // scale_w)):
            cv2.line(thumbnail, (i, 0), (i, small_h), (255, 255, 255), 1)

        cv2.rectangle(
            thumbnail,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            5,
        )

        st.write("### Select a block")

        coordinates = streamlit_image_coordinates.streamlit_image_coordinates(
            thumbnail, use_column_width=True
        )

    if coordinates:
        which_x = math.floor((w * coordinates["x"] / coordinates["width"]) / bx)
        which_y = math.floor((h * coordinates["y"] / coordinates["height"]) / by)
        if which_x != st.session_state.get("which_x", 0):
            st.session_state["which_x"] = which_x
        if which_y != st.session_state.get("which_y", 0):
            st.session_state["which_y"] = which_y

        st.rerun()

    # image = blocks[which_y, which_x, 0]
    with st.sidebar:
        st.image(image, caption="Selected block", use_container_width=True)

    st.session_state.model_name = model_name
    st.session_state.image_inference = image
    st.session_state.image_id = (
        (file_id + str(which_x) + str(which_y) + str(bx) + str(by) + str(model_name))
        + ("_use_tta" if use_tta else "_no_tta")
        + ("_detect_errors" if detect_errors else "_no_detect_errors")
    )
    col1, col2, col3 = st.columns([1, 1, 1])

    if download_button:
        st.success("Infering on the full segmented image...")
        prediction = ui_inference_cacheless(
            _model=model_name if not use_ensemble else "ensemble",
            _image=org_image,
            _device="cuda" if torch.cuda.is_available() else "cpu",
            use_tta=use_tta,
            use_correction=detect_errors,
            prediction_threshold=prediction_threshold,
            pixel_size=st.session_state.get("pixel_size", 0.13),
            verbose=False,
        )
        org_h, org_w = org_image.shape[:2]
        labelmap = prediction.valid_copy().get_labelmap(org_h, org_w, 3)
        labelmap = (CMAP(labelmap)[:, :, :3] * 255).astype(np.uint8)

        image = Image.fromarray(labelmap)

        # Create an in-memory binary stream
        buffer = io.BytesIO()

        # Save the image to the buffer in PNG format
        image.save(buffer, format="jpeg")

        # Get the byte data from the buffer
        jpeg_data = buffer.getvalue()

        st.download_button(
            "Download full segmented image",
            data=jpeg_data,
            file_name="full_segmented_image.jpg",
            mime="image/jpeg",
        )

    start_inference(
        use_tta=use_tta,
        use_correction=detect_errors,
        prediction_threshold=prediction_threshold,
    )


else:
    st.switch_page("pages/1_Load.py")

# Add a callback to mouse move event

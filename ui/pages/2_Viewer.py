import streamlit as st
from bokeh.plotting import figure
from bokeh.layouts import gridplot
from streamlit_bokeh import streamlit_bokeh
from dnafiber.ui.utils import (
    get_image,
    get_multifile_image,
    get_resized_image,
    bokeh_imshow,
    pad_image_to_croppable,
)
from dnafiber.deployment import MODELS_ZOO
from dnafiber.ui.inference import ui_inference
from skimage.util import view_as_blocks
import cv2
import math
from bokeh.models import (
    Range1d,
    HoverTool,
)
import streamlit_image_coordinates
from catppuccin import PALETTE
import numpy as np
import torch
from skimage.segmentation import expand_labels
from dnafiber.deployment import _get_model

st.set_page_config(layout="wide")
st.title("Viewer")


@st.cache_resource
def get_model(model_name):
    model = _get_model(
        device="cuda" if torch.cuda.is_available() else "cpu",
        revision=model_name,
    )
    return model


def start_inference():
    image = st.session_state.image_inference
    model = get_model(st.session_state.model)
    prediction = ui_inference(
        model,
        image,
        "cuda" if torch.cuda.is_available() else "cpu",
        st.session_state.post_process,
        st.session_state.image_id,
    )
    prediction = [
        p
        for p in prediction
        if (p.fiber_type != "single") and p.fiber_type != "multiple"
    ]
    max_width = 2048
    if image.shape[1] > max_width:
        st.toast("Images are displayed at a lower resolution of 2048 pixel wide")

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
    for fiber in prediction:
        fiber.bbox = (
            int(fiber.bbox[0] * scale),
            int(fiber.bbox[1] * scale),
            int(fiber.bbox[2] * scale),
            int(fiber.bbox[3] * scale),
        )
        fiber.data = cv2.resize(
            expand_labels(fiber.data, 1),
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_NEAREST,
        )
    h, w = image.shape[:2]
    labels_maps = np.zeros((h, w), dtype=np.uint8)
    for i, region in enumerate(prediction):
        h, w = region.data.shape[:2]
        labels_maps[
            region.bbox[1] : region.bbox[1] + h,
            region.bbox[0] : region.bbox[0] + w,
        ] = region.data
    p1 = figure(
        width=600,
        x_range=Range1d(-image.shape[1] / 8, image.shape[1] * 1.125, bounds="auto"),
        y_range=Range1d(image.shape[0] * 1.125, -image.shape[0] / 8, bounds="auto"),
        title=f"Detected fibers: {len(prediction)}",
        tools="pan,wheel_zoom,box_zoom,reset",
        active_scroll="wheel_zoom",
    )

    p1.image(
        image=[labels_maps],
        x=0,
        y=0,
        dw=labels_maps.shape[1],
        dh=labels_maps.shape[0],
        palette=["black", st.session_state["color1"], st.session_state["color2"]] if np.max(labels_maps) > 0 else ["black"],
    )
    p2 = figure(
        x_range=p1.x_range,
        y_range=p1.y_range,
        width=600,
        tools="pan,wheel_zoom,box_zoom,reset",
        active_scroll="wheel_zoom",
    )
    bokeh_imshow(p2, image)
    colors = [c.hex for c in PALETTE.latte.colors][:14]
    data_source = dict(
        x=[], y=[], width=[], height=[], color=[], firstAnalog=[], secondAnalog=[], ratio=[]
    )
    np.random.shuffle(colors)
    for i, region in enumerate(prediction):
        color = colors[i % len(colors)]
        x, y, w, h = region.bbox

        data_source["x"].append((x + w / 2))
        data_source["y"].append((y + h / 2))
        data_source["width"].append(w)
        data_source["height"].append(h)
        data_source["color"].append(color)
        r, g = region.counts
        red_length = st.session_state["pixel_size"] * r / scale
        green_length = st.session_state["pixel_size"] * g / scale
        data_source["firstAnalog"].append(f"{red_length:.2f} µm")
        data_source["secondAnalog"].append(f"{green_length:.2f} µm")
        data_source["ratio"].append(f"{green_length / red_length:.2f}")

    rect1 = p1.rect(
        x="x",
        y="y",
        width="width",
        height="height",
        source=data_source,
        fill_color=None,
        line_color="color",
    )
    rect2 = p2.rect(
        x="x",
        y="y",
        width="width",
        height="height",
        source=data_source,
        fill_color=None,
        line_color="color",
    )

    hover = HoverTool(
        tooltips=f'<p style="color:{st.session_state["color1"]};">@firstAnalog</p> <p style="color:{st.session_state["color2"]};">@secondAnalog</p><b> Ratio: @ratio</b>',
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
    streamlit_bokeh(fig, use_container_width=True)


def on_session_start():
    can_start = st.session_state.get("files_uploaded", None) is not None and len(st.session_state.files_uploaded) > 0

    if can_start:
        return can_start
    
    cldu_exists = st.session_state.get("files_uploaded_cldu", None) is not None and len(st.session_state.files_uploaded_cldu) > 0
    idu_exists = st.session_state.get("files_uploaded_idu", None) is not None and len(st.session_state.files_uploaded_idu) > 0

    if cldu_exists and idu_exists:
        
        if len(st.session_state.get("files_uploaded_cldu")) != len(st.session_state.get("files_uploaded_idu")):
            st.error("Please upload the same number of CldU and IdU files.")
            return False

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
    selected_file = st.selectbox("Pick an image", displayed_names, index=0, help="Select an image to view and analyze.")
    
    # Find index of the selected file
    index = displayed_names.index(selected_file)
    file = files[index]
    if isinstance(file, tuple):
        file_id = file[0].file_id if file[0] is not None else file[1].file_id
        if file[0] is None or file:
            missing = "First analog" if file[0] is None else "Second analog"
            st.warning("In this image, {missing} channel is missing. We assume the intended goal is to segment the DNA fibers without differentiation. \
                       Note the model may still predict two classes and try to compute a ratio; these informations can be ignored.")
        image = get_multifile_image(file)
    else:
        file_id = file.file_id
        image = get_image(file, reverse_channel=st.session_state.get("reverse_channels", False),  id=file_id)
    h, w = image.shape[:2]

    with st.sidebar:
        st.metric(
            "Pixel size (µm)",
            st.session_state.get("pixel_size", 0.13),
        )

        block_size = st.slider(
            "Block size",
            min_value=256,
            max_value=min(4096, max(h, w)),
            value=min(2048, max(h, w)),
            step=256,
        )
    if h < block_size:
        block_size = h
    if w < block_size:
        block_size = w

    bx = by = block_size
    image = pad_image_to_croppable(image, bx, by, file_id + str(bx) + str(by))
    thumbnail = get_resized_image(image, file_id)

    blocks = view_as_blocks(image, (bx, by, 3))
    x_blocks, y_blocks = blocks.shape[0], blocks.shape[1]
    with st.sidebar:
        with st.expander("Model", expanded=True):
            model_name = st.selectbox(
                "Select a model",
                list(MODELS_ZOO.keys()),
                index=0,
                help="Select a model to use for inference",
            )
            finetuned = st.checkbox(
                "Use finetuned model",
                value=True,
                help="Use a finetuned model for inference",
            )

            col1, col2 = st.columns(2)
            with col1:
                st.write("Running on:")
            with col2:
                st.button(
                    "GPU" if torch.cuda.is_available() else "CPU",
                    disabled=True,
                )

            st.session_state.post_process = st.checkbox(
                "Post-process",
                value=True,
                help="Apply post-processing to the prediction",
            )

        st.session_state.model = (
            (MODELS_ZOO[model_name] + "_finetuned")
            if finetuned
            else MODELS_ZOO[model_name]
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

    image = blocks[which_y, which_x, 0]
    with st.sidebar:
        st.image(image, caption="Selected block", use_container_width=True)

    st.session_state.image_inference = image
    st.session_state.image_id = (
        file_id + str(which_x) + str(which_y) + str(bx) + str(by)
    )
    col1, col2, col3 = st.columns([1, 1, 1])
    start_inference()
else:
    st.switch_page("pages/1_Load.py")

# Add a callback to mouse move event

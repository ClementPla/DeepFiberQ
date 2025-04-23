import streamlit as st
from bokeh.plotting import figure
from bokeh.layouts import gridplot
from streamlit_bokeh import streamlit_bokeh
from dnafiber.ui.utils import (
    get_image,
    get_resized_image,
    bokeh_imshow,
    pad_image_to_croppable,
    MODELS_ZOO,
)
from dnafiber.ui.inference import inference
from skimage.util import view_as_blocks
import cv2
import math
from bokeh.models import (
    Range1d,
    HoverTool,
)
import streamlit_image_coordinates
from skimage.measure import regionprops, label
from catppuccin import PALETTE
import numpy as np
import torch

st.set_page_config(layout="wide")
st.title("Visualization")


def start_inference():
    image = st.session_state.image_inference
    prediction = inference(
        st.session_state.model,
        image,
        "cuda" if torch.cuda.is_available() else "cpu",
        st.session_state.image_id,
    )
    # Scale the image and the prediction to max_width
    max_width = 2048
    if image.shape[1] > max_width:
        st.toast("Images are displayed at a lower resolution of 2048 pixel wide")

    image = cv2.resize(
        image,
        (max_width, int(max_width * image.shape[0] / image.shape[1])),
        interpolation=cv2.INTER_LINEAR,
    )

    scale = max_width / prediction.shape[1]
    labels = label(prediction > 0)
    rprops = regionprops(labels)
    prediction = cv2.resize(
        prediction,
        (max_width, int(max_width * prediction.shape[0] / prediction.shape[1])),
        interpolation=cv2.INTER_NEAREST_EXACT,
    )
    labels = cv2.resize(
        labels,
        (max_width, int(max_width * labels.shape[0] / labels.shape[1])),
        interpolation=cv2.INTER_NEAREST_EXACT,
    )

    # prediction = expand_labels(prediction, 1)

    p1 = figure(
        width=600,
        x_range=Range1d(-image.shape[1] / 8, image.shape[1] * 1.125, bounds="auto"),
        y_range=Range1d(image.shape[0] * 1.125, -image.shape[0] / 8, bounds="auto"),
        title=f"Detected fibers: {len(rprops)}",
        tools="pan,wheel_zoom,box_zoom,reset",
        active_scroll="wheel_zoom",
    )

    p1.image(
        image=[prediction],
        x=0,
        y=0,
        dw=prediction.shape[1],
        dh=prediction.shape[0],
        palette=["black", "red", "green"] if np.max(prediction) > 0 else ["black"],
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
        x=[], y=[], width=[], height=[], color=[], red=[], green=[], ratio=[]
    )
    np.random.shuffle(colors)
    for i, region in enumerate(rprops):
        color = colors[i % len(colors)]
        minr, minc, maxr, maxc = region.bbox
        minr = minr * scale
        minc = minc * scale
        maxr = maxr * scale
        maxc = maxc * scale
        data_source["x"].append((minc + maxc) / 2)
        data_source["y"].append((minr + maxr) / 2)
        data_source["width"].append(maxc - minc)
        data_source["height"].append(maxr - minr)
        data_source["color"].append(color)
        red_length = (
            st.session_state["pixel_size"]
            * np.sum(((labels == region.label) * prediction) == 1)
            / scale
        )
        green_length = (
            st.session_state["pixel_size"]
            * np.sum(((labels == region.label) * prediction) == 2)
            / scale
        )
        data_source["red"].append(f"{red_length:.2f} µm")
        data_source["green"].append(f"{green_length:.2f} µm")
        data_source["ratio"].append(f"{red_length / green_length:.2f}")

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
        tooltips='<p style="color:red;">@red</p> <p style="color:green;">@green</p><b> Ratio: @ratio</b>',
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


if st.session_state.get("files_uploaded", None):
    files = st.session_state.files_uploaded
    selected_file = st.selectbox("Pick an image", [f.name for f in files])
    file = [f for f in files if f.name == selected_file][-1]
    image = get_image(file, file.file_id)
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
    image = pad_image_to_croppable(image, bx, by, file.file_id + str(bx) + str(by))
    thumbnail = get_resized_image(image, file.file_id)

    blocks = view_as_blocks(image, (bx, by, 3))
    x_blocks, y_blocks = blocks.shape[0], blocks.shape[1]
    with st.sidebar:
        with st.expander("Model"):
            model_name = st.selectbox(
                "Select a model",
                list(MODELS_ZOO.keys()),
                index=0,
                help="Select a model to use for inference",
            )
            col1, col2 = st.columns(2)
            with col1:
                st.write("Running on:")
            with col2:
                st.button(
                    "GPU" if torch.cuda.is_available() else "CPU",
                    disabled=True,
                )

        st.session_state.model = MODELS_ZOO[model_name]

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
        file.file_id + str(which_x) + str(which_y) + str(bx) + str(by)
    )
    col1, col2, col3 = st.columns([1, 1, 1])
    start_inference()
else:
    st.switch_page("pages/1_Load.py")

# Add a callback to mouse move event

import streamlit as st
from bokeh.plotting import figure
from bokeh.layouts import row, grid, gridplot
from streamlit_bokeh import streamlit_bokeh
from skimage.segmentation import expand_labels
from ui.utils import get_image, get_resized_image, bokeh_imshow
from ui.inference import inference
from skimage.util import view_as_blocks
import cv2
import math
from bokeh.models import Range1d
import streamlit_image_coordinates
from skimage.measure import regionprops, label
from catppuccin import PALETTE
import numpy as np

st.set_page_config(layout="wide")
st.title("Visualization")


def start_inference():
    image = st.session_state.image_inference
    prediction = inference(image, "cuda", st.session_state.image_id)

    # Scale the image and the prediction to max_width
    max_width = 1024
    image = cv2.resize(
        image,
        (max_width, int(max_width * image.shape[0] / image.shape[1])),
        interpolation=cv2.INTER_NEAREST,
    )
    prediction = cv2.resize(
        prediction,
        (max_width, int(max_width * prediction.shape[0] / prediction.shape[1])),
        interpolation=cv2.INTER_NEAREST,
    )

    rprops = regionprops(label(prediction > 0))

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
        palette=["black", "red", "green"],
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
    np.random.shuffle(colors)
    for i, region in enumerate(rprops):
        color = colors[i % len(colors)]
        minr, minc, maxr, maxc = region.bbox
        p1.rect(
            x=(minc + maxc) / 2,
            y=(minr + maxr) / 2,
            width=maxc - minc + 10,
            height=maxr - minr + 10,
            line_color=color,
            fill_color=None,
            line_width=2,
        )
        p2.rect(
            x=(minc + maxc) / 2,
            y=(minr + maxr) / 2,
            width=maxc - minc + 10,
            height=maxr - minr + 10,
            line_color=color,
            fill_color=None,
            line_width=2,
        )
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
    selected_file = st.selectbox("Pick one", [f.name for f in files])
    file = [f for f in files if f.name == selected_file][-1]
    image = get_image(file, file.file_id)
    with st.sidebar:
        block_size = st.slider(
            "Block size",
            min_value=256,
            max_value=2048,
            value=2048,
            step=256,
        )
    bx = by = block_size
    # Pad the image to be divisible by bx and by
    h, w = image.shape[:2]
    if h % bx != 0:
        pad_h = bx - (h % bx)
    else:
        pad_h = 0
    if w % by != 0:
        pad_w = by - (w % by)
    else:
        pad_w = 0
    image = cv2.copyMakeBorder(
        image,
        math.ceil(pad_h / 2),
        math.floor(pad_h / 2),
        math.ceil(pad_w / 2),
        math.floor(pad_w / 2),
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )
    thumbnail = get_resized_image(image, file.file_id)

    blocks = view_as_blocks(image, (bx, by, 3))
    x_blocks, y_blocks = blocks.shape[0], blocks.shape[1]
    with st.sidebar:
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
        st.session_state["which_x"] = which_x
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

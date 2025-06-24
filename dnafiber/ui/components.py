import streamlit as st
import cv2
import pandas as pd
from dnafiber.ui.utils import numpy_to_base64_png
import math

@st.cache_data
def show_fibers(_prediction, _image, image_id=None):
    data = dict(
        fiber_id=[],
        firstAnalog=[],
        secondAnalog=[],
        ratio=[],
        fiber_type=[],
        visualization=[],
    )

    for fiber in _prediction:
        data["fiber_id"].append(fiber.fiber_id)
        r, g = fiber.counts
        red_length = st.session_state["pixel_size"] * r
        green_length = st.session_state["pixel_size"] * g
        data["firstAnalog"].append(f"{red_length:.3f} ")
        data["secondAnalog"].append(f"{green_length:.3f} ")
        data["ratio"].append(f"{green_length / red_length:.3f}")
        data["fiber_type"].append(fiber.fiber_type)

        x, y, w, h = fiber.bbox

        # Offset by half the height and width the x, y coordinates to have a larger visualization
        offsetX = int(w)
        offsetY = int(h)
        x = max(0, x - offsetX)
        y = max(0, y - offsetY)
        w += offsetX * 2
        h += offsetY * 2
        visu = _image[y : y + h, x : x + w, :]
        visu = cv2.normalize(visu, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # Scale the visualization to a minimum width of 256 pixels
        
        if visu.shape[1] != 400:
            scale = 400 / visu.shape[1]
            visu = cv2.resize(
                visu,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_LINEAR,
            )
            offsetX = math.floor(offsetX * scale)
            offsetY = math.floor(offsetY * scale)

        # Draw a rectangle around the fiber, without the offset
        cv2.rectangle(
            visu,
            (offsetX, offsetY),
            (visu.shape[1] - offsetX, visu.shape[0] - offsetY),
            (0, 0, 255),
            2,
        )

        # Make sure the 
        data["visualization"].append(visu)

    df = pd.DataFrame(data)
    df = df.rename(
        columns={
            "firstAnalog": "First analog (µm)",
            "secondAnalog": "Second analog (µm)",
            "ratio": "Ratio",
            "fiber_type": "Fiber type",
            "fiber_id": "Fiber ID",
            "visualization": "Visualization",
        }
    )
    df["Visualization"] = df["Visualization"].apply(lambda x: numpy_to_base64_png(x))
    return df


def table_components(df):
    event = st.dataframe(
            df,
            on_select="rerun",
            selection_mode="multi-row",
            use_container_width=True,
            column_config={
                "Visualization": st.column_config.ImageColumn(
                    "Visualization",
                    help="Visualization of the fiber",
                )
            },
        )

    rows = event["selection"]["rows"]
    columns = df.columns[:-2]
    df = df.iloc[rows][columns]

    cols = st.columns(3)
    with cols[0]:
        copy_to_clipboard = st.button(
            "Copy selected fibers to clipboard",
            help="Copy the selected fibers to clipboard in CSV format.",
        )
        if copy_to_clipboard:
            df.to_clipboard(index=False)
    with cols[2]:

        if st.session_state.get("image_id", None) is not None:
             st.download_button(
                "Download selected fibers",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name=f"fibers_{st.session_state.image_id}.csv",
                mime="text/csv",
            )
        else:
            st.download_button(
                "Download selected fibers",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="fibers_segment.csv",
                mime="text/csv",
            )
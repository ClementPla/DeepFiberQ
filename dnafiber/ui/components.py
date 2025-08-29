import math

import cv2
import pandas as pd
import streamlit as st

from dnafiber.data.utils import numpy_to_base64_jpeg


@st.cache_data
def show_fibers(_prediction, _image, image_id=None, resolution=400, show_errors=True):
    return show_fibers_cacheless(_prediction, _image, image_id=image_id, resolution=resolution, show_errors=show_errors)

def show_fibers_cacheless(_prediction, _image, image_id=None, resolution=400, show_errors=True):
    data = dict(
        fiber_id=[],
        firstAnalog=[],
        secondAnalog=[],
        ratio=[],
        fiber_type=[],
        visualization=[],
        is_valid=[],
    )

    for fiber in _prediction:
        if not show_errors and not fiber.is_valid:
            continue
        data["fiber_id"].append(fiber.fiber_id)
        r, g = fiber.counts
        red_length = st.session_state["pixel_size"] * r
        green_length = st.session_state["pixel_size"] * g
        data["firstAnalog"].append(f"{red_length:.3f} ")
        data["secondAnalog"].append(f"{green_length:.3f} ")
        data["ratio"].append(f"{green_length / red_length:.3f}")
        data["fiber_type"].append(fiber.fiber_type)
        data["is_valid"].append(fiber.is_valid)

        x, y, w, h = fiber.bbox
      
        # Offset by half the height and width the x, y coordinates to have a larger visualization
        xextract1 = max(0, x-math.floor(w / 2))
        yextract1 = max(0, y-math.floor(h / 2))
        xextract2 = min(_image.shape[1], x + w + math.floor(w / 2))
        yextract2 = min(_image.shape[0], y + h + math.floor(h / 2))

        visu = _image[max(0, yextract1) : min(_image.shape[0], yextract2), max(0, xextract1) : min(_image.shape[1], xextract2)]

        rect_coordinates = (x - xextract1, y - yextract1, w, h)

        visu = cv2.normalize(visu, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        # Draw the bbox on the visualization
        # Scale the visualization to a minimum width or height of `resolution` pixels

        max_dim = max(visu.shape[0], visu.shape[1])
        if max_dim < resolution:
            scale = resolution / max_dim
            visu = cv2.resize(
                visu,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_LINEAR,
            )

            # Scale the rectangle coordinates as well
            rect_coordinates = (
                math.floor(rect_coordinates[0] * scale),
                math.floor(rect_coordinates[1] * scale),
                math.floor(rect_coordinates[2] * scale),
                math.floor(rect_coordinates[3] * scale),
            )

        cv2.rectangle(visu, (rect_coordinates[0], rect_coordinates[1]), (rect_coordinates[0] + rect_coordinates[2], rect_coordinates[1] + rect_coordinates[3]), (0, 0, 255), 3)

        # Draw a rectangle around the fiber, without the offset
        
        # Pad the visualization to have a square image
        if visu.shape[0] < visu.shape[1]:
            pad = (visu.shape[1] - visu.shape[0]) // 2
            visu = cv2.copyMakeBorder(
                visu, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )
        elif visu.shape[0] > visu.shape[1]:
            pad = (visu.shape[0] - visu.shape[1]) // 2
            visu = cv2.copyMakeBorder(
                visu, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0)
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
    df["Visualization"] = df["Visualization"].apply(lambda x: numpy_to_base64_jpeg(x))
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
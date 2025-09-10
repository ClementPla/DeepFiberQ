import math

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from dnafiber.data.utils import numpy_to_base64_jpeg
from dnafiber.postprocess.fiber import Fibers


import plotly.express as px


@st.cache_data
def show_fibers(
    _prediction, _image, inference_id=None, resolution=400, show_errors=True
):
    return show_fibers_cacheless(
        _prediction,
        _image,
        resolution=resolution,
        show_errors=show_errors,
    )


def show_fibers_cacheless(_prediction, _image, resolution=400, show_errors=True):
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
        data["fiber_id"].append(fiber.fiber_id)
        r, g = fiber.counts
        red_length = st.session_state["pixel_size"] * r
        green_length = st.session_state["pixel_size"] * g
        data["firstAnalog"].append(f"{red_length:.3f} ")
        data["secondAnalog"].append(f"{green_length:.3f} ")
        data["ratio"].append(
            f"{green_length / red_length if red_length > 0 else 0:.3f}"
        )
        data["fiber_type"].append(fiber.fiber_type)
        data["is_valid"].append(fiber.is_acceptable)

        x, y, w, h = fiber.bbox

        # Offset by half the height and width the x, y coordinates to have a larger visualization
        xextract1 = max(0, x - math.floor(w / 2))
        yextract1 = max(0, y - math.floor(h / 2))
        xextract2 = min(_image.shape[1], x + w + math.floor(w / 2))
        yextract2 = min(_image.shape[0], y + h + math.floor(h / 2))
        visu = _image[
            max(0, yextract1) : min(_image.shape[0], yextract2),
            max(0, xextract1) : min(_image.shape[1], xextract2),
        ]

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

        cv2.rectangle(
            visu,
            (rect_coordinates[0], rect_coordinates[1]),
            (
                rect_coordinates[0] + rect_coordinates[2],
                rect_coordinates[1] + rect_coordinates[3],
            ),
            (0, 0, 255),
            3,
        )

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
    columns = df.columns
    selected_df = df.iloc[rows][columns]

    cols = st.columns(3)
    with cols[0]:
        copy_to_clipboard = st.button(
            "Copy selected fibers to clipboard",
            help="Copy the selected fibers to clipboard in CSV format.",
        )
        if copy_to_clipboard:
            selected_df.to_clipboard(index=False)
    with cols[1]:
        st.download_button(
            "Download valid fibers",
            data=df[df["is_valid"]].to_csv(index=False).encode("utf-8"),
            file_name=f"fibers_valid.csv",
            mime="text/csv",
        )
    with cols[2]:
        st.download_button(
            "Download selected fibers",
            data=selected_df.to_csv(index=False).encode("utf-8"),
            file_name="fibers_segment.csv",
            mime="text/csv",
        )


def distribution_analysis(predictions: Fibers):
    df = predictions.to_df()
    df = df[df["Fiber type"] == "double"]
    df = df[(df.Ratio > 0.125) & (df.Ratio < 8)]
    df["Length"] = df["First analog (µm)"] + df["Second analog (µm)"]
    cap = st.checkbox(
        "Cap number of fibers",
        value=False,
        help="If checked, we only keep the N fibers closest to the barycenter of the distribution.",
    )
    if cap:
        N = st.slider(
            "Number of fibers",
            min_value=1,
            max_value=len(df),
            value=50,
            step=1,
        )
    cap_values = ["Length"]
    mean_points = df[cap_values].median().values

    df["Distance"] = np.linalg.norm(
        df[cap_values].values - mean_points,
        axis=1,
    )

    col1, col2 = st.columns(2)
    with col1:
        fig = px.box(
            df.nsmallest(N if cap else len(df), "Distance"),
            y="Ratio",
            points="all",
            title="Distribution of the Ratio",
            labels={"Ratio": "Ratio (second analog / first analog)"},
        )
        fig.update_layout(height=500, margin=dict(l=0, r=0, b=0, t=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure(
            data=go.Scatter3d(
                x=df["First analog (µm)"],
                y=df["Second analog (µm)"],
                z=df["Length"],
                mode="markers",
                marker=dict(size=1, color=df["Distance"]),
            )
        )
        # Draw a radius around the barycenter that contains N points
        # if cap:
        #     radius = df.nsmallest(N, "Distance").Distance.max()

        #     # Create a sphere
        #     u = np.linspace(0, 2 * np.pi, 25)
        #     v = np.linspace(0, np.pi, 25)
        #     x = mean_points[0] + radius * np.outer(np.cos(u), np.sin(v))
        #     y = mean_points[1] + radius * np.outer(np.sin(u), np.sin(v))
        #     z = mean_points[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
        #     fig.add_trace(
        #         go.Surface(
        #             x=x,
        #             y=y,
        #             z=z,
        #             opacity=0.2,
        #             showscale=False,
        #         )
        #     )

        # fig.add_trace(
        #     go.Scatter3d(
        #         x=[mean_points[0]],
        #         y=[mean_points[1]],
        #         z=[mean_points[2]],
        #         mode="markers",
        #         marker=dict(size=5, color="green"),
        #         name="Barycenter",
        #     )
        # )
        # Set axis labels
        fig.update_layout(
            scene=dict(
                xaxis_title="First analog (µm)",
                yaxis_title="Second analog (µm)",
                zaxis_title="Length (µm)",
            ),
            height=700,
        )
        st.plotly_chart(fig, use_container_width=True)


@st.cache_data(max_entries=5)
def viewer_components(_image, _prediction, inference_id):
    image = _image
    if image.max() > 25:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        max_size = 10000
        h, w = image.shape[:2]
        size = max(h, w)
        scale = 1.0
        if size > max_size:
            scale = max_size / size
            image = cv2.resize(
                image,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_LINEAR,
            )

        return image, scale

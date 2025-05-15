import streamlit as st
from dnafiber.inference import infer
from dnafiber.inference import post_process
from skimage.util import apply_parallel
from skimage.measure import label, regionprops
import torch
from dnafiber.ui.utils import get_image, MODELS_ZOO, TYPE_MAPPING
import numpy as np
import pandas as pd
import plotly.express as px


def _parallel_process(image):
    return post_process(image[:, :, 0])


def plot_result():
    if st.session_state.get("results", None) is None:
        return
    only_bilateral = st.checkbox(
        "Show only bilateral fibers",
        value=False,
        help="Show only fibers with a ratio of 1",
    )
    min_ratio, max_ratio = st.slider(
        "Ratio range",
        min_value=0.0,
        max_value=10.0,
        value=(0.0, 5.0),
        step=0.1,
        help="Select the ratio range to display",
    )
    df = st.session_state.results.copy()

    clean_df = df[["ratio", "image_name", "fiber_type"]].copy()
    clean_df["Image"] = clean_df["image_name"]
    clean_df["Fiber Type"] = clean_df["fiber_type"]
    clean_df["Ratio"] = clean_df["ratio"]
    if only_bilateral:
        clean_df = clean_df[clean_df["Fiber Type"] == "BILATERAL"]
    clean_df = clean_df[
        (clean_df["Ratio"] >= min_ratio) & (clean_df["Ratio"] <= max_ratio)
    ]

    fig = px.violin(
        clean_df,
        y="Ratio",
        x="Image",
        color="Image",
        box=True,  # draw box plot inside the violin
        points="all",  # can be 'outliers', or False
    )
    st.plotly_chart(
        fig,
        use_container_width=True,
    )


def compute_properties(predictions, pixel_size, image_name):
    # Placeholder for actual property computation
    # This function should compute properties based on the predictions and pixel size
    fibers_map = predictions[:, :, 0] > 0
    fibers_type = predictions[:, :, 1]
    labeled_fibers = label(fibers_map)

    properties = regionprops(labeled_fibers, intensity_image=predictions[:, :, 0])
    results = dict(red=[], green=[], length=[], ratio=[], image_name=[], fiber_type=[])

    for prop in properties:
        fiber_label = prop.label
        fiber_image = prop.image_intensity
        red = np.sum(fiber_image == 1) * pixel_size
        green = np.sum(fiber_image == 2) * pixel_size
        if red == 0 or green == 0:
            continue
        length = red + green
        ratio = green / red
        fiber_type = TYPE_MAPPING[
            int(np.max(fibers_type[labeled_fibers == fiber_label]))
        ]
        if red == 0 or green == 0:
            continue
        results["red"].append(red)
        results["green"].append(green)
        results["length"].append(length)
        results["ratio"].append(ratio)
        results["image_name"].append(image_name.split("-")[0])
        results["fiber_type"].append(fiber_type)
    return results


def run_inference(model_name, pixel_size):
    my_bar = st.progress(0, text="Running segmentation...")
    all_files = st.session_state.files_uploaded
    all_results = dict(
        red=[], green=[], length=[], ratio=[], image_name=[], fiber_type=[]
    )
    for i, file in enumerate(all_files):
        image = get_image(file, file.file_id)

        prediction = infer(
            model_name,
            image,
            "cuda",
            scale=pixel_size,
        )

        refined = np.stack([prediction, np.zeros_like(prediction)], axis=-1)

        predictions = apply_parallel(
            _parallel_process, refined, chunks=(2048, 2048, 2), dtype="uint8"
        )
        results = compute_properties(predictions, pixel_size, file.name)
        for k, v in all_results.items():
            all_results[k].extend(results[k])
        my_bar.progress(i / len(all_files), text=f"{file.name} done")

    st.session_state.results = pd.DataFrame.from_dict(all_results)

    my_bar.empty()


if st.session_state.get("files_uploaded", None):
    run_segmentation = st.button("Run Segmentation")

    with st.sidebar:
        pixel_size = st.number_input(
            "Pixel size (um/pixel)",
            min_value=0.0,
            max_value=100.0,
            value=st.session_state.get("pixel_size", 0.13),
            step=0.1,
            help="Pixel size in micrometers per pixel",
        )
        with st.expander("Model", expanded=True):
            model_name = st.selectbox(
                "Select a model",
                list(MODELS_ZOO.keys()),
                index=0,
                help="Select a model to use for inference",
            )
            finetuned = st.checkbox(
                "Use finetuned model",
                value=False,
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

    tab_segmentation, tab_charts = st.tabs(["Segmentation", "Charts"])
    if run_segmentation:
        run_inference(
            model_name=MODELS_ZOO[model_name] + "_finetuned"
            if finetuned
            else MODELS_ZOO[model_name],
            pixel_size=pixel_size,
        )
    with tab_segmentation:
        st.subheader("Segmentation")
        if run_segmentation:
            st.write(
                st.session_state.results,
                use_container_width=True,
                hide_index=True,
            )
            st.balloons()
            st.download_button(
                label="Download results",
                data=st.session_state.results.to_csv(index=False).encode("utf-8"),
                file_name="results.csv",
                mime="text/csv",
            )
    with tab_charts:
        plot_result()

else:
    st.switch_page("pages/1_Load.py")

import streamlit as st
import torch
from dnafiber.ui.utils import get_multifile_image, get_image_cacheless
from dnafiber.deployment import MODELS_ZOO
import pandas as pd
import plotly.express as px
from dnafiber.postprocess import refine_segmentation
import torch.nn.functional as F
from joblib import Parallel, delayed
import time
from catppuccin import PALETTE
from dnafiber.deployment import _get_model
from dnafiber.ui.inference import ui_inference_cacheless
from dnafiber.ui.components import show_fibers, table_components


def plot_result(seleted_category):
    if st.session_state.get("results", None) is None or selected_category is None:
        return
    only_bilateral = st.checkbox(
        "Show only bicolor fibers",
        value=False,
    )
    remove_outliers = st.checkbox(
        "Remove outliers",
        value=True,
        help="Remove outliers from the data",
    )
    reorder = st.checkbox(
        "Reorder groups by median ratio",
        value=True,
    )
    if remove_outliers:
        min_ratio, max_ratio = st.slider(
            "Ratio range",
            min_value=0.0,
            max_value=10.0,
            value=(0.0, 5.0),
            step=0.1,
            help="Select the ratio range to display",
        )
    df = st.session_state.results.copy()

    clean_df = df[["Ratio", "image_name", "Fiber type"]].copy()
    clean_df["Ratio"] = clean_df["Ratio"].astype(float)
    clean_df["Image"] = clean_df["image_name"]

    if only_bilateral:
        clean_df = clean_df[clean_df["Fiber type"] == "double"]
    if remove_outliers:
        clean_df = clean_df[
            (clean_df["Ratio"] >= min_ratio) & (clean_df["Ratio"] <= max_ratio)
        ]

    if selected_category:
        clean_df = clean_df[clean_df["Image"].isin(selected_category)]

        if not reorder:
            clean_df["Image"] = pd.Categorical(
                clean_df["Image"], categories=selected_category, ordered=True
            )
            clean_df.sort_values("Image", inplace=True)

    if reorder:
        image_order = (
            clean_df.groupby("Image")["Ratio"]
            .median()
            .sort_values(ascending=True)
            .index
        )
        clean_df["Image"] = pd.Categorical(
            clean_df["Image"], categories=image_order, ordered=True
        )
        clean_df.sort_values("Image", inplace=True)

    palette = [c.hex for c in PALETTE.latte.colors]

    fig = px.violin(
        clean_df,
        y="Ratio",
        x="Image",
        color="Image",
        box=True,  # draw box plot inside the violin
        points="all",  # can be 'outliers', or False
        color_discrete_sequence=palette,
    )
    # Set y-axis to log scale
    st.plotly_chart(
        fig,
        use_container_width=True,
    )


def run_inference(model_name, pixel_size):
    is_cuda_available = torch.cuda.is_available()
    if "ensemble" in model_name:
        model = [
            _ + "_finetuned" if "finetuned" in model_name else ""
            for _ in MODELS_ZOO.values()
            if _ != "ensemble"
        ]
    else:
        model = _get_model(
            revision=model_name,
            device="cuda" if is_cuda_available else "cpu",
        )

    my_bar = st.progress(0, text="Running segmentation...")
    all_files = st.session_state.files_uploaded
    all_results = []
    for i, file in enumerate(all_files):
        if isinstance(file, tuple):
            if file[0] is None:
                filename = file[1].name
            if file[1] is None:
                filename = file[0].name
            image = get_multifile_image(file)
        else:
            filename = file.name
            image = get_image_cacheless(
                file, st.session_state.get("reverse_channels", False), file.file_id
            )
        start = time.time()
        prediction = ui_inference_cacheless(
            _model=model,
            _image=image,
            _device="cuda" if is_cuda_available else "cpu",
            postprocess=False,
            only_segmentation=True,
        )
        print(f"Prediction time: {time.time() - start:.2f} seconds for {file.name}")
        h, w = prediction.shape
        start = time.time()
        y_size, x_size = 8192, 8192
        if h > y_size or w > x_size:
            # Extract blocks from the prediction

            blocks = [
                (image[y:y+y_size, x: x+x_size], prediction[y : y + y_size, x : x + x_size], y, x)
                for y in range(0, h, y_size)
                for x in range(0, w, x_size)
            ]
            

            parallel_results = Parallel(n_jobs=4, verbose=10)(
                delayed(refine_segmentation)(
                    block_img,
                    block,
                    True,
                    st.session_state.get("elongation_threshold", 2),
                    x,
                    y,
                )
                for (block_img, block, y, x) in (
                    blocks
                )
            )

            results = [
                fiber for block_result in parallel_results for fiber in block_result
            ]
        else:
            results = refine_segmentation(
                image,
                prediction,
                post_process=True,
                threshold=st.session_state.get("elongation_threshold", 50),
            )
        print(f"Refinement time: {time.time() - start:.2f} seconds for {filename}")
        results = [fiber for fiber in results if fiber.is_valid]
        df = show_fibers(results, image, image_id=None)
        df["image_name"] = filename
        all_results.append(df)
        my_bar.progress(i / len(all_files), text=f"{filename} done")

    # Create a dictionary to store the results by concatenating the results
    results_dict = {k: [] for k in all_results[0].keys()}
    for result in all_results:
        for k, v in result.items():
            results_dict[k].extend(v)

    my_bar.empty()
    st.session_state.results = pd.DataFrame(results_dict)


if st.session_state.get("files_uploaded", None):
    run_segmentation = st.button("Run Segmentation", use_container_width=True)

    with st.sidebar:
        st.metric(
            "Pixel size (µm)",
            st.session_state.get("pixel_size", 0.13),
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

    tab_segmentation, tab_charts = st.tabs(["Segmentation", "Charts"])

    with tab_segmentation:
        st.subheader("Segmentation")
        if run_segmentation:
            st.session_state.image_id = None
            run_inference(
                model_name=MODELS_ZOO[model_name] + "_finetuned"
                if finetuned
                else MODELS_ZOO[model_name],
                pixel_size=st.session_state.get("pixel_size", 0.13),
            )
            st.balloons()
        if st.session_state.get("results", None) is not None:
            table_components(st.session_state.results)

    with tab_charts:
        if st.session_state.get("results", None) is not None:
            results = st.session_state.results

            categories = results["image_name"].unique()
            selected_category = st.multiselect(
                "Select a category", categories, default=categories
            )
            plot_result(selected_category)

else:
    st.switch_page("pages/1_Load.py")

import streamlit as st
import torch
from dnafiber.ui.consts import DefaultValues
from dnafiber.ui.utils import build_file_id, get_multifile_image, get_image_cacheless
from dnafiber.deployment import MODELS_ZOO, MODELS_ZOO_R, ENSEMBLE, Models
import pandas as pd
import plotly.express as px

from catppuccin import PALETTE
from dnafiber.ui.inference import ui_inference, get_model
from dnafiber.ui.components import show_fibers_cacheless, table_components
from dnafiber.ui.utils import build_inference_id
from dnafiber.ui import DefaultValues as DV

from dnafiber.ui.utils import retain_session_state

retain_session_state(st.session_state)


def image_name_to_category(image_name):
    """
    Convert image name to category.
    This function assumes that the image name is in the format 'category-image_name'.
    """

    return "-".join(image_name.split("-")[:-1])


def plot_result(selected_category):
    if st.session_state.get("results", None) is None or selected_category is None:
        return
    only_bilateral = st.checkbox(
        "Show only bicolor fibers",
        value=True,
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

    show_points = st.checkbox(
        "Show points",
        value=True,
        help="Show a swarm plot next to the distribution plot",
    )
    normalize = st.checkbox(
        "Normalize ratios",
        value=False,
        help="Normalize ratios to a baseline",
    )
    baseline = None
    if normalize:
        baseline = st.selectbox(
            "Select baseline",
            options=st.session_state.results["image_name"]
            .apply(image_name_to_category)
            .unique(),
            help="Select the baseline to normalize ratios",
        )

    if remove_outliers:
        min_ratio, max_ratio = st.slider(
            "Ratio range",
            min_value=0.0,
            max_value=10.0,
            value=(0.0, 10.0),
            step=0.1,
            help="Select the ratio range to display",
        )
    df = st.session_state.results.copy()

    clean_df = df[["Ratio", "image_name", "Fiber type"]].copy()
    clean_df["Ratio"] = clean_df["Ratio"].astype(float)
    clean_df["Image"] = clean_df["image_name"].apply(image_name_to_category)

    if only_bilateral:
        clean_df = clean_df[clean_df["Fiber type"] == "double"]
    if remove_outliers:
        clean_df = clean_df[
            (clean_df["Ratio"] >= min_ratio) & (clean_df["Ratio"] <= max_ratio)
        ]
    if baseline:
        mean_value_baseline = clean_df[clean_df["Image"] == baseline]["Ratio"].median()
        clean_df["Ratio"] = clean_df["Ratio"] / mean_value_baseline

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

    fig = px.box(
        clean_df,
        y="Ratio",
        x="Image",
        color="Image",
        points="all" if show_points else "outliers",  # can be 'outliers', or False
        color_discrete_sequence=palette,
        log_y=True,  # Set y-axis to log scale
        range_y=[0.125 / 2, 16],
    )

    # Set y-axis ticks to log2 scale
    fig.update_yaxes(
        tickvals=[0.25, 0.5, 1, 2, 4, 8],
        ticktext=["0.25", "0.5", "1", "2", "4", "8"],
        type="log",
    )
    # Set y-axis to log scale
    st.plotly_chart(
        fig,
        use_container_width=True,
    )


def run_one_file(
    file,
    model,
    use_tta=DV.USE_TTA,
    use_correction=DV.USE_CORRECTION,
    prediction_threshold=DV.PREDICTION_THRESHOLD,
    inference_id="",
):
    if isinstance(file, tuple):
        if file[0] is None:
            filename = file[1].name
        if file[1] is None:
            filename = file[0].name
        image = get_multifile_image(
            file, bit_depth=st.session_state.get("bit_depth", DV.BIT_DEPTH)
        )
    else:
        image = get_image_cacheless(
            file,
            st.session_state.get("reverse_channels", DV.REVERSE_CHANNELS),
            st.session_state.get("bit_depth", DV.BIT_DEPTH),
        )
        filename = file.name
    results = ui_inference(
        model,
        image,
        "cuda" if torch.cuda.is_available() else "cpu",
        use_tta,
        use_correction,
        prediction_threshold,
        key=inference_id,
    )
    results = results.filtered_copy()
    st.write(f"Detected {len(results)} fibers in {filename}")
    df = show_fibers_cacheless(results, image, resolution=256)
    df["image_name"] = filename
    return df


def run_inference(model_name, use_tta=DV.USE_TTA, use_correction=DV.USE_CORRECTION):
    if model_name == Models.ENSEMBLE:
        model = []
        for _ in range(len(ENSEMBLE)):
            with st.spinner(f"Loading model {_ + 1}/{len(ENSEMBLE)}..."):
                model.append(get_model(ENSEMBLE[_]))
    else:
        with st.spinner("Loading model..."):
            model = get_model(model_name)

    my_bar = st.progress(0, text="Running segmentation...")
    all_files = st.session_state.files_uploaded
    all_results = []
    for i, file in enumerate(all_files):
        if isinstance(file, tuple):
            if file[0] is None:
                filename = file[1].name
            if file[1] is None:
                filename = file[0].name
        else:
            filename = file.name
        file_id = build_file_id(
            file,
            pixel_size=st.session_state.get("pixel_size", DV.PIXEL_SIZE),
            reverse_channels=st.session_state.get(
                "reverse_channels", DV.REVERSE_CHANNELS
            ),
            bit_depth=st.session_state.get("bit_depth", DV.BIT_DEPTH),
        )
        prediction_threshold = st.session_state.get(
            "prediction_threshold", DV.PREDICTION_THRESHOLD
        )

        inference_id = build_inference_id(
            file_id,
            model_name,
            use_tta,
            use_correction,
            prediction_threshold,
        )
        try:
            df = run_one_file(
                file,
                model,
                use_tta,
                use_correction,
                inference_id=inference_id,
                prediction_threshold=prediction_threshold,
            )
        except Exception as e:
            st.error(f"Error processing {filename}: {e}")
            continue
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
            st.session_state.get("pixel_size", DV.PIXEL_SIZE),
        )

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
                disabled=st.session_state.get("use_ensemble", False),
            )
            if st.session_state.get("use_ensemble", False):
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

    tab_segmentation, tab_charts = st.tabs(["Segmentation", "Charts"])

    with tab_segmentation:
        st.subheader("Segmentation")
        if run_segmentation:
            run_inference(
                model_name=model_name,
                use_tta=st.session_state.get("use_tta", False),
                use_correction=st.session_state.get("use_correction", False),
            )
            st.balloons()
        if st.session_state.get("results", None) is not None:
            table_components(st.session_state.results)

    with tab_charts:
        if st.session_state.get("results", None) is not None:
            results = st.session_state.results

            categories = (
                results["image_name"].apply(image_name_to_category).unique().tolist()
            )
            selected_category = st.multiselect(
                "Select a category", categories, default=categories
            )
            plot_result(selected_category)

else:
    st.switch_page("pages/1_Load.py")

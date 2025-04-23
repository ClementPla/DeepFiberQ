import streamlit as st
from dnafiber.ui.inference import inference
from dnafiber.ui.utils import (
    get_image,
    get_resized_image,
    bokeh_imshow,
    pad_image_to_croppable,
)


if st.session_state.get("files_uploaded", None):
    tab_segmentation, tab_charts = st.tabs(["Segmentation", "Charts"])
    with tab_segmentation:
        st.subheader("Segmentation")
        run_segmentation = st.button("Run Segmentation")
        if run_segmentation:
            pass

    # for f in st.session_state.files_uploaded:
    #     pass
else:
    st.switch_page("pages/1_Load.py")


def run_inference():
    for file in st.session_state.files_uploaded:
        image = get_image(file, file.file_id)

        prediction = inference(image, "cuda", file.file_id + "_all_file")

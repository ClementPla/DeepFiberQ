import streamlit as st

st.session_state["files_uploaded"] = st.file_uploader(
    label="Upload a file",
    accept_multiple_files=True,
    type=["czi", "jpeg", "jpg", "png", "tiff", "tif"],
)


st.session_state["pixel_size"] = st.number_input(
    "Please confirm the pixel size of the image in µm (default: 0.13 µm). Make sure all the images have the same pixel size.",
    value=0.13,
)

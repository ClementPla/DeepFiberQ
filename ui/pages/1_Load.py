import streamlit as st

st.set_page_config(
    page_title="DeepFiberQ++",
)
if (
    st.session_state.get("files_uploaded", None) is None
    or len(st.session_state.files_uploaded) == 0
):
    st.session_state["files_uploaded"] = st.file_uploader(
        label="Upload a file",
        accept_multiple_files=True,
        type=["czi", "jpeg", "jpg", "png", "tiff", "tif"],
    )
else:
    st.warning(
        "To upload new files, please refresh the page. Note that new files will replace the current ones."
    )

cols = st.columns(3)
with cols[0]:
    st.write("### Channel interpretation")
    st.write(
        "Please select the channel to be used for the analysis (only for CZI files)."
    )
with cols[2]:
    st.write("### Pixel size")
    st.session_state["pixel_size"] = st.number_input(
        "Please confirm the pixel size of the image in µm (default: 0.13 µm).",
        value=0.13,
    )
    # In small, lets precise the tehnical details
    st.write(
        "The pixel size is used to convert the pixel coordinates to µm. The model is trained on images with a pixel size of 0.26 µm, and the application automatically resizes the images to this pixel size using your provided choice."
    )

import streamlit as st


st.session_state["files_uploaded"] = st.file_uploader(
    label="Upload a file",
    accept_multiple_files=True,
    type=["czi"],
)

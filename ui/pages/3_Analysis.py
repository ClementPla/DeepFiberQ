import streamlit as st
from ui.inference import inference

if st.session_state.get("files_uploaded", None):
    for f in st.session_state.files_uploaded:
        pass
else:
    st.switch_page("pages/1_Load.py")

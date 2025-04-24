import streamlit as st
import torch


def main():
    st.set_page_config(
        page_title="Hello",
        page_icon="🧬",
        layout="wide",
    )
    st.write("# Welcome to DeepFiberQ++! 👋")

    st.write(
        "This is a web application for the DeepFiberQ++ project, which aims to provide an easy-to-use interface for analyzing and processing fiber images."
    )
    st.write("## Features")
    st.write(
        "- **Image loading**: The application accepts CZI file, jpeg and PNG file. \n"
        "- **Image segmentation**: The application provides a set of tools to segment the DNA fiber and measure the ratio. \n"
    )
    st.write("## Technical details")
    cols = st.columns(2)
    with cols[0]:
        st.write("### Source")
        st.write("The source code for this application is available on GitHub.")
        """
        [![Repo](https://badgen.net/badge/icon/GitHub?icon=github&label)](https://github.com/ClementPla/DeepFiberQ/tree/relabelled) 

        """
        st.markdown("<br>", unsafe_allow_html=True)

    with cols[1]:
        st.write("### Device ")
        st.write("If available, the application will try to use a GPU for processing.")
        device = "GPU" if torch.cuda.is_available() else "CPU"
        cols = st.columns(3)
        with cols[0]:
            st.write("Running on:")
        with cols[1]:
            st.button(device, icon="⚙️", disabled=True)
        if not torch.cuda.is_available():
            with cols[2]:
                st.warning("The application will run on CPU, which may be slower.")


if __name__ == "__main__":
    main()

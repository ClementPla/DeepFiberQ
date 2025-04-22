import streamlit as st


def main():
    st.set_page_config(
        page_title="Hello",
        page_icon="👋",
    )
    st.write("# Welcome to DeepFiberQ++! 👋")

    st.session_state["pixel_size"] = st.number_input(
        "Please enter the pixel size of the image in µm (default: 0.13 µm)",
        value=0.13,
    )


if __name__ == "__main__":
    main()

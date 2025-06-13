import streamlit as st


st.set_page_config(
    page_title="DeepFiberQ++",
    page_icon=":microscope:",
    layout="wide",
)

def build_multichannel_loader():

    if (
        st.session_state.get("files_uploaded", None) is None
        or len(st.session_state.files_uploaded) == 0
    ):
        st.session_state["files_uploaded"] = st.file_uploader(
            label="Upload files",
            accept_multiple_files=True,
            type=["czi", "jpeg", "jpg", "png", "tiff", "tif"],
        )
    else:
        st.session_state["files_uploaded"] += st.file_uploader(
            label="Upload files",
            accept_multiple_files=True,
            type=["czi", "jpeg", "jpg", "png", "tiff", "tif"],
        )
    st.write("### Channel interpretation")
    st.markdown("The goal is to obtain an RGB image in the order of <span style='color: red;'>First analog</span>, <span style='color: green;'>Second analog</span>, <span style='color: blue;'>Empty</span>.", unsafe_allow_html=True)
    st.markdown("By default, we assume that the first channel in CZI/TIFF file is <span style='color: green;'>the second analog</span>, (which happens to be the case in Zeiss microscope) " \
    "which means that we swap the order of the two channels for processing.", unsafe_allow_html=True)
    st.write("If this not the intented behavior, please tick the box below:")
    st.session_state["reverse_channels"] = st.checkbox(
        "Reverse the channels interpretation",
        value=False,
    )
    st.warning("Please note that we only swap the channels for raw (CZI, TIFF) files. JPEG and PNG files "\
               "are assumed to be already in the correct order (First analog in red and second analog in green). " \
    )

    st.info("" \
    "The channels order in CZI files does not necessarily match the order in which they are displayed in ImageJ or equivalent. " \
    "Indeed, such viewers will usually look at the metadata of the file to determine the order of the channels, which we don't. " \
    "In doubt, we recommend visualizing the image in ImageJ and compare with our viewer. If the channels appear reversed, tick the option above.")

def build_individual_loader():
   
    cols = st.columns(2)
    with cols[1]:
        st.markdown(f"<h3 style='color: {st.session_state["color2"]};'>Second analog</h3>", unsafe_allow_html=True)
        cldu_files = st.file_uploader(
            label="Upload second analog file(s)",
            accept_multiple_files=True,
            type=["czi", "jpeg", "jpg", "png", "tiff", "tif"],
        )
    with cols[0]:
        st.markdown(f"<h3 style='color: {st.session_state["color1"]};'>First analog</h3>", unsafe_allow_html=True)
        idu_files = st.file_uploader(
            label="Upload first analog file(s)",
            accept_multiple_files=True,
            type=["czi", "jpeg", "jpg", "png", "tiff", "tif"],
        )
    
    if idu_files is None and cldu_files is None:
        return 
    else:
        # Check we have both IdU and CldU files (same number of files)
        if len(idu_files)>0 and len(cldu_files)>0 and len(idu_files) != len(cldu_files):
            st.error("Please upload the same number of IdU and CldU files.")
            return
    idu_files = sorted(idu_files, key=lambda x: x.name)
    cldu_files = sorted(cldu_files, key=lambda x: x.name)
    max_size = max(len(idu_files), len(cldu_files))

    # Pad the shorter list with None
    if len(idu_files) < max_size:
        idu_files += [None] * (max_size - len(idu_files))
    if len(cldu_files) < max_size:
        cldu_files += [None] * (max_size - len(cldu_files))

    combined_files = list(zip(idu_files, cldu_files))
    if (
        st.session_state.get("files_uploaded", None) is None
        or len(st.session_state.files_uploaded) == 0
    ):
        st.session_state["files_uploaded"] = combined_files
    else:
        st.session_state["files_uploaded"] += combined_files




cols = st.columns(2)
with cols[1]:
    

    st.write("### Pixel size")
    st.session_state["pixel_size"] = st.number_input(
        "Please indicate the pixel size of the image in µm (default: 0.13 µm).",
        value=st.session_state.get("pixel_size", 0.13),
    )
    # In small, lets precise the tehnical details
    st.write(
        "The pixel size is used to convert the pixel coordinates to µm. " \
        "The model is trained on images with a pixel size of 0.26 µm, and the application automatically " \
        "resizes the images to match this pixel size using your provided choice."
    )

    st.write("### Labels color")
    color_choices = st.columns(2)
    with color_choices[0]:
        st.session_state["color1"] = st.color_picker(
            "Select the color for first analog",
            value=st.session_state.get("color1", "#FF0000"),
            help="This color will be used to display the first analog segments.")
    with color_choices[1]:
        st.session_state["color2"] = st.color_picker(
            "Select the color for second analog",
            value=st.session_state.get("color2", "#00FF00"),
            help="This color will be used to display the second analog segments.")

with cols[0]:
    choice = st.segmented_control(
        "Please select the type of images you want to upload:",
        options=["Multichannel", "Individual channel"],
        default="Multichannel",
    )
    if choice == "Individual channel":
        build_individual_loader()
    else:
        build_multichannel_loader()



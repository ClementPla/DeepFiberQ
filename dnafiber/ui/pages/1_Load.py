import streamlit as st


st.set_page_config(
    page_title="DN-AI",
    page_icon="🔬",
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
        st.markdown(f"<h3 style='color: {st.session_state['color2']};'>Second analog</h3>", unsafe_allow_html=True)

        if (
            st.session_state.get("analog_2_files", None) is None
            or len(st.session_state.analog_2_files) == 0
        ):
            st.session_state["analog_2_files"] = st.file_uploader(
                label="Upload second analog file(s)",
                accept_multiple_files=True,
                type=["czi", "jpeg", "jpg", "png", "tiff", "tif"],
            )
        else:
            st.session_state["analog_2_files"] += st.file_uploader(
                label="Upload second analog file(s)",
                accept_multiple_files=True,
                type=["czi", "jpeg", "jpg", "png", "tiff", "tif"],
            )
        
        
    with cols[0]:
        st.markdown(f"<h3 style='color: {st.session_state['color1']};'>First analog</h3>", unsafe_allow_html=True)
        if (
            st.session_state.get("analog_1_files", None) is None
            or len(st.session_state.analog_1_files) == 0
        ):
            st.session_state["analog_1_files"] = st.file_uploader(
                label="Upload first analog file(s)",
                accept_multiple_files=True,
                type=["czi", "jpeg", "jpg", "png", "tiff", "tif"],
            )
        else:
            st.session_state["analog_1_files"] += st.file_uploader(
                label="Upload first analog file(s)",
                accept_multiple_files=True,
                type=["czi", "jpeg", "jpg", "png", "tiff", "tif"],)
    
    analog_1_files=st.session_state.get("analog_1_files", None)
    analog_2_files=st.session_state.get("analog_2_files", None)
    
    # Remove duplicates from the list of files. We loop through the files and keep only the first occurrence of each file_id.
    def remove_duplicates(files):
        seen_ids = set()
        unique_files = []
        for file in files:
            if file and file.name not in seen_ids:
                unique_files.append(file)
                seen_ids.add(file.name)
        return unique_files

    analog_1_files = remove_duplicates(analog_1_files or [])
    analog_2_files = remove_duplicates(analog_2_files or [])
    
    
    if analog_1_files is None and analog_2_files is None:
        return 
    else:
        if len(analog_1_files)>0 and len(analog_2_files)>0 and len(analog_1_files) != len(analog_2_files):
            st.error("Please upload the same number of analogs files.")
            return
    
    # Always make sure we don't have duplicates in the list of files
    
    analog_1_files = sorted(analog_1_files, key=lambda x: x.name)
    analog_2_files = sorted(analog_2_files, key=lambda x: x.name)
    max_size = max(len(analog_1_files), len(analog_2_files))
    # Pad the shorter list with None
    if len(analog_1_files) < max_size:
        analog_1_files += [None] * (max_size - len(analog_1_files))
    if len(analog_2_files) < max_size:
        analog_2_files += [None] * (max_size - len(analog_2_files))

    combined_files = list(zip(analog_1_files, analog_2_files))

    

    if (
        st.session_state.get("files_uploaded", None) is None
        or len(st.session_state.files_uploaded) == 0
    ):
        st.session_state["files_uploaded"] = combined_files
    else:
        st.session_state["files_uploaded"] += combined_files
    


    # If any of the files (analog_1_files or analog_2_files) was included previously in the files_uploaded, 
    # We remove the previous occurence from the files_uploaded list.
    current_ids = set()
    for f in analog_1_files + analog_2_files:
        if f:
            current_ids.add(f.name)

    # Safely filter the list to exclude any files with matching file_ids
    def is_not_duplicate(file):
        if isinstance(file, tuple):
            f1, f2 = file
            if f1 and f2:
                return True
            
            return (f1 is None or f1.name not in current_ids) and (f2 is None or f2.name not in current_ids)
        else:
            return True
        
    st.session_state.files_uploaded = [f for f in st.session_state.files_uploaded if is_not_duplicate(f)]



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



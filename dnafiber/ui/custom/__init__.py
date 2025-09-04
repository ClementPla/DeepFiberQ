import os
from turtle import st
import streamlit.components.v1 as components
import streamlit as st
from dnafiber.data.utils import numpy_to_base64_jpeg

_RELEASE = True


if not _RELEASE:
    _component_func = components.declare_component(
        # We give the component a simple, descriptive name ("my_component"
        # does not fit this bill, so please choose something better for your
        # own component :)
        "fiber_ui",
        # Pass `url` here to tell Streamlit that the component will be served
        # by the local dev server that you run via `npm run start`.
        # (This is useful while your component is in development.)
        url="http://localhost:3001",
    )
else:
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(parent_dir, "frontend/build")
    _component_func = components.declare_component("fiber_ui", path=build_dir)


def fiber_ui(image, fibers, key=None):
    """Create a new instance of "fiber_ui".

    Parameters
    ----------

    Returns
    -------

    """

    # convert image to base64
    with st.spinner("Preparing image..."):
        data_uri = numpy_to_base64_jpeg(image)
    component_value = _component_func(
        image=data_uri,
        elements=fibers,
        image_w=image.shape[1],
        image_h=image.shape[0],
        key=key,
        default=False,
    )
    return component_value

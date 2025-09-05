from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError


from dnafiber.ui.consts import DefaultValues
from dnafiber.ui.utils import init_session_states

init_session_states()

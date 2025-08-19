import attrs
import numpy as np
from typing import Tuple
from dnafiber.postprocess.skan import trace_skeleton

@attrs.define
class Bbox:
    x: int
    y: int
    width: int
    height: int

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)

    @bbox.setter
    def bbox(self, value: Tuple[int, int, int, int]):
        self.x, self.y, self.width, self.height = value


@attrs.define
class Fiber:
    bbox: Bbox
    data: np.ndarray


@attrs.define
class FiberProps:
    fiber: Fiber
    fiber_id: int
    red_pixels: int = None
    green_pixels: int = None
    category: str = None
    is_valid: bool = True

    @property
    def bbox(self):
        return self.fiber.bbox.bbox

    @bbox.setter
    def bbox(self, value):
        self.fiber.bbox = value

    @property
    def data(self):
        return self.fiber.data

    @data.setter
    def data(self, value):
        self.fiber.data = value

    @property
    def red(self):
        if self.red_pixels is None:
            self.red_pixels, self.green_pixels = self.counts
        return self.red_pixels

    @property
    def green(self):
        if self.green_pixels is None:
            self.red_pixels, self.green_pixels = self.counts
        return self.green_pixels

    @property
    def length(self):
        return sum(self.counts)

    @property
    def counts(self):
        if self.red_pixels is None or self.green_pixels is None:
            self.red_pixels = np.sum(self.data == 1)
            self.green_pixels = np.sum(self.data == 2)
        return self.red_pixels, self.green_pixels

    @property
    def fiber_type(self):
        if self.category is not None:
            return self.category
        red_pixels, green_pixels = self.counts
        if red_pixels == 0 or green_pixels == 0:
            self.category = "single"
        else:
            self.category = estimate_fiber_category(self.data)
        return self.category

    @property
    def ratio(self):
        return self.green / self.red

    @property
    def is_valid(self):

        try:
            fiber_type = self.fiber_type
        except IndexError:
            # Happens if there is no pixel remaining for this fiber, which indicates it is invalid.
            return False
        
        return (
            fiber_type == "double"
            or fiber_type == "one-two-one"
            or fiber_type == "two-one-two"
        )
    
    def scaled_coordinates(self, scale: float) -> Tuple[int, int]:
        """
        Scale down the coordinates of the fiber's bounding box.
        """
        x, y, width, height = self.bbox
        return (
            int(x * scale),
            int(y * scale),
            int(width * scale),
            int(height * scale),
        )


def estimate_fiber_category(fiber: np.ndarray) -> str:
    """
    Estimate the fiber category based on the number of red and green pixels.
    """
    coordinates = trace_skeleton(fiber > 0)
    coordinates = np.asarray(coordinates)
    try:
        values = fiber[coordinates[:, 0], coordinates[:, 1]]
    except IndexError:
        return "unknown"
    diff = np.diff(values)
    jump = np.sum(diff != 0)
    n_ccs = jump + 1
    if n_ccs == 2:
        return "double"
    elif n_ccs == 3:
        if values[0] == 1:
            return "one-two-one"
        else:
            return "two-one-two"
    else:
        return "multiple"

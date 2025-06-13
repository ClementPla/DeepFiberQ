import attrs
import numpy as np
from typing import Tuple
from dnafiber.postprocess.skan import trace_skeleton


@attrs.define
class Fiber:
    bbox: Tuple[int, int, int, int]
    data: np.ndarray


@attrs.define
class FiberProps:
    fiber: Fiber
    fiber_id: int
    red_pixels: int = None
    green_pixels: int = None
    category: str = None

    @property
    def bbox(self):
        return self.fiber.bbox

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
        return (
            self.fiber_type == "double"
            or self.fiber_type == "one-two-one"
            or self.fiber_type == "two-one-two"
        )


def estimate_fiber_category(fiber: np.ndarray) -> str:
    """
    Estimate the fiber category based on the number of red and green pixels.
    """
    coordinates = trace_skeleton(fiber > 0)
    coordinates = np.asarray(coordinates)
    values = fiber[coordinates[:, 0], coordinates[:, 1]]
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

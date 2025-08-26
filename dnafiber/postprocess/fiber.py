import attrs
import numpy as np
from typing import Optional, Tuple
from dnafiber.postprocess.skan import trace_skeleton
from skimage.segmentation import expand_labels


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
    is_an_error: bool = False

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
        if self.red == 0:
            return np.nan
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
        ) and not self.is_an_error

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

    def bbox_intersect(self, other: Fiber, ratio=0.25) -> bool:
        """
        Check if the bounding boxes of two fibers intersect by at least a certain ratio.
        """
        x1, y1, w1, h1 = self.bbox
        x2, y2, w2, h2 = other.bbox

        intersection_area = max(0, min(x1 + w1, x2 + w2) - max(x1, x2)) * max(
            0, min(y1 + h1, y2 + h2) - max(y1, y2)
        )
        self_area = w1 * h1
        other_area = w2 * h2
        return (
            intersection_area / float(self_area + other_area - intersection_area)
            >= ratio
        )


@attrs.define
class Fibers:
    fibers: list[FiberProps]

    def __iter__(self):
        return iter(self.fibers)

    def __getitem__(self, index):
        return self.fibers[index]

    def __len__(self):
        return len(self.fibers)

    @property
    def ratios(self):
        return [fiber.ratio for fiber in self.fibers]

    @property
    def reds(self):
        return [fiber.red for fiber in self.fibers]

    @property
    def greens(self):
        return [fiber.green for fiber in self.fibers]

    def get_labelmap(self, h, w, fiber_width=1):
        labelmap = np.zeros((h, w), dtype=np.uint8)
        for fiber in self.fibers:
            x, y, w, h = fiber.bbox
            roi = labelmap[y : y + h, x : x + w]
            binary = fiber.data > 0
            roi[binary] = fiber.data[binary]

        labelmap = expand_labels(labelmap, fiber_width)
        return labelmap

    def contains(self, fiber: Fiber, ratio=0.5):
        for existing_fiber in self.fibers:
            if existing_fiber.bbox_intersect(fiber, ratio):
                return True
        return False

    def append(self, fiber: Fiber):
        self.fibers.append(fiber)

    def append_if_not_exists(self, fiber: Fiber, ratio=0.5):
        """
        Append a fiber to the list if it does not already exist.
        """
        if not self.contains(fiber, ratio):
            self.append(fiber)

    def valid_copy(self):
        return Fibers([fiber for fiber in self.fibers if fiber.is_valid])

    def union(self, other, ratio=0.5):
        union = Fibers(self.fibers)
        for fiber in other:
            union.append_if_not_exists(fiber, ratio)
        return union

    def difference(self, other, ratio):
        substract = Fibers([])
        intersection = self.intersection(other, ratio)
        for fiber in self.fibers:
            if not intersection.contains(fiber, ratio):
                substract.append_if_not_exists(fiber, ratio)
        return substract

    def intersection(self, other, ratio=0.5):
        intersection = Fibers([])
        for fiber in self.fibers:
            for other_fiber in other:
                if fiber.bbox_intersect(other_fiber, ratio):
                    intersection.append_if_not_exists(fiber, ratio)
        return intersection

    def to_df(
        self, pixel_size=0.13, img_name: Optional[str] = None, filter_invalid=True
    ):
        import pandas as pd

        data = {
            "Fiber ID": [],
            "First analog (µm)": [],
            "Second analog (µm)": [],
            "Ratio": [],
            "Fiber type": [],
            "Valid": [],
        }

        for i, fiber in enumerate(self.fibers):
            if filter_invalid and not fiber.is_valid:
                continue
            data["Fiber ID"].append(i)
            r, g = fiber.counts
            red_length = pixel_size * r
            green_length = pixel_size * g
            data["First analog (µm)"].append(red_length)
            data["Second analog (µm)"].append(green_length)
            data["Ratio"].append(fiber.ratio)
            data["Fiber type"].append(fiber.fiber_type)
            data["Valid"].append(fiber.is_valid)
        df = pd.DataFrame.from_dict(data)
        if img_name:
            df["Image Name"] = img_name
        return df


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

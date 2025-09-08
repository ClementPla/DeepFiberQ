import attrs
import numpy as np
from typing import Optional, Tuple
from dnafiber.postprocess.skan import trace_skeleton
from skimage.segmentation import expand_labels
from dnafiber.postprocess.utils import generate_svg


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

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)

    def to_dict(self):
        return {
            "x": int(self.x),
            "y": int(self.y),
            "width": int(self.width),
            "height": int(self.height),
        }

    def __getitem__(self, index):
        return self.bbox[index]


@attrs.define
class FiberProps:
    bbox: Bbox
    data: np.ndarray
    fiber_id: int = -1
    red_pixels: int = None
    green_pixels: int = None
    category: str = None
    is_an_error: bool = False
    svg_rep: str = None  # SVG representation of the fiber, for visualization purposes
    trace: np.ndarray = None  # Coordinates of the skeletons of the fiber

    @property
    def bbox(self):
        return self.bbox.bbox

    @bbox.setter
    def bbox(self, value):
        self.bbox = value

    @property
    def data(self):
        return self.data

    @data.setter
    def data(self, value):
        self.data = value

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
            self.category = estimate_fiber_category(self.get_trace(), self.data)
        return self.category

    def get_trace(self):
        if self.trace is not None:
            return self.trace
        # Generate trace if not provided
        self.trace = np.asarray(trace_skeleton(self.data > 0))
        if not self.trace.size:
            self.trace = np.empty((0, 2), dtype=int)
        return self.trace

    @property
    def ratio(self):
        if self.red == 0:
            return np.nan
        return self.green / self.red

    @property
    def is_valid(self):
        try:
            _ = self.fiber_type
        except IndexError:
            # Happens if there is no pixel remaining for this fiber, which indicates it is invalid.
            return False

        return self.is_double or self.is_triple

    @property
    def is_acceptable(self):
        return not self.is_an_error

    @property
    def is_double(self):
        return self.fiber_type == "double"

    @property
    def is_triple(self):
        return self.fiber_type in ["one-two-one", "two-one-two"]

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

    def bbox_intersect(self, other, ratio=0.25) -> bool:
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

    def svg_representation(self, scale=1.0):
        try:
            svg_representation = generate_svg(self, scale=scale)
        except Exception as e:
            print(f"Error generating SVG representation: {e}")
            return None
        return svg_representation


def filter_invalid_bbox(fibers: list[FiberProps]) -> list[FiberProps]:
    valid_fibers = []
    for fiber in fibers:
        x, y, w, h = fiber.bbox
        if w > 0 and h > 0 and fiber.data.size > 0 and x >= 0 and y >= 0:
            valid_fibers.append(fiber)
    return valid_fibers


@attrs.define
class Fibers:
    fibers: list[FiberProps] = attrs.field(factory=list, converter=filter_invalid_bbox)

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
    def lengths(self):
        return [fiber.length for fiber in self.fibers]

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

    def contains(self, fiber: FiberProps, ratio=0.5):
        for existing_fiber in self.fibers:
            if existing_fiber.bbox_intersect(fiber, ratio):
                return True
        return False

    def append(self, fiber: FiberProps):
        self.fibers.append(fiber)

    def append_if_not_exists(self, fiber: FiberProps, ratio=0.5):
        """
        Append a fiber to the list if it does not already exist.
        """
        if not self.contains(fiber, ratio):
            self.append(fiber)

    def valid_copy(self):
        return Fibers([fiber for fiber in self.fibers if fiber.is_valid])

    def filtered_copy(self):
        return Fibers(
            [fiber for fiber in self.fibers if (fiber.is_acceptable and fiber.is_valid)]
        )

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

    def svgs(self, scale=1.0):
        svgs = [fiber.svg_representation(scale) for fiber in self.fibers]
        return [svg for svg in svgs if svg is not None]

    def debug(self, h=1024, w=1024):
        import matplotlib.pyplot as plt
        from dnafiber.data.utils import CMAP

        labelmap = self.get_labelmap(h, w)

        ratio = np.mean(self.ratios)
        plt.imshow(labelmap, cmap=CMAP)
        plt.title(f"Fiber Labelmap, {len(self.fibers)} fibers, mean ratio: {ratio:.2f}")
        plt.axis("off")
        plt.show()


def estimate_fiber_category(fiber_trace: np.ndarray, fiber_data: np.ndarray) -> str:
    """
    Estimate the fiber category based on the number of red and green pixels.
    """
    coordinates = fiber_trace
    coordinates = np.asarray(coordinates)
    try:
        values = fiber_data[coordinates[:, 0], coordinates[:, 1]]
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

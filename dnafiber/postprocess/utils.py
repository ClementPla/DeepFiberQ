from __future__ import annotations

import json
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dnafiber.postprocess.fiber import FiberProps


def generate_svg(fiber: FiberProps, scale=1.0) -> str:
    """Generate an SVG representation of the fiber for visualization purposes.
    Parameters
    ----------
    fiber : Fiber
        The fiber object containing the data to be visualized.
    scale : float, optional
        Scaling factor for the SVG coordinates, by default 1.0
    """
    # Placeholder implementation; replace with actual SVG generation logic

    bbox_data = fiber.bbox.to_dict()
    trace_data = fiber.get_trace()

    offset_x, offset_y = bbox_data["x"], bbox_data["y"]
    data = fiber.data[trace_data[:, 0], trace_data[:, 1]]

    traces_polylines = []
    current_color = data[0]
    current_line = [(trace_data[0, 1], trace_data[0, 0])]
    colors = []
    for j, (color, x, y) in enumerate(zip(data, trace_data[:, 1], trace_data[:, 0])):
        if color != current_color:
            # Close the previous path
            current_line.append((x, y))
            traces_polylines.append(
                " ".join(
                    [
                        f"{int((x + offset_x) * scale)},{int((y + offset_y) * scale)}"
                        for x, y in current_line
                    ]
                )
            )
            colors.append("red" if current_color == 1 else "green")
            current_color = color
            current_line = [(x, y)]
        # Only append the new x, y coordinates if the distance to the previous one is greater to 2 pixels
        if (
            current_line
            and ((x - current_line[-1][0]) ** 2 + (y - current_line[-1][1]) ** 2) ** 0.5
            > 5
            and (j != len(data) - 1)
            and j != 0
        ):
            current_line.append((x, y))

    traces_polylines.append(
        " ".join(
            [
                f"{int((x + offset_x) * scale)},{int((y + offset_y) * scale)}"
                for x, y in current_line
            ]
        )
    )
    colors.append("red" if color == 1 else "green")

    bbox_data["points"] = traces_polylines
    bbox_data["colors"] = colors
    bbox_data["x"] = int(bbox_data["x"] * scale)
    bbox_data["y"] = int(bbox_data["y"] * scale)
    bbox_data["width"] = int(bbox_data["width"] * scale)
    bbox_data["height"] = int(bbox_data["height"] * scale)
    bbox_data["id"] = fiber.fiber_id
    bbox_data["type"] = fiber.fiber_type
    bbox_data["ratio"] = fiber.ratio if not np.isnan(fiber.ratio) else -1
    bbox_data["is_error"] = bool(fiber.is_an_error[0])
    return json.dumps(bbox_data)

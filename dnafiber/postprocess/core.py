import numpy as np
import cv2
from typing import List, Tuple
from dnafiber.postprocess.skan import find_endpoints, compute_points_angle
from scipy.spatial.distance import cdist

from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_array
from skimage.morphology import skeletonize
from dnafiber.postprocess.skan import find_line_intersection, prolongate_endpoints
from dnafiber.postprocess.fiber import Fiber, FiberProps, Bbox, Fibers
from itertools import compress
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from dnafiber.postprocess.error_detection import correct_fibers

cmlabel = ListedColormap(["black", "red", "green"])

MIN_ANGLE = 20
MIN_BRANCH_LENGTH = 10
MIN_BRANCH_DISTANCE = 30


def handle_multiple_fiber_in_cc(fiber, junctions_fiber, coordinates):
    for y, x in junctions_fiber:
        fiber[y - 1 : y + 2, x - 1 : x + 2] = 0

    endpoints = find_endpoints(fiber > 0)
    endpoints = np.asarray(endpoints)
    # We only keep the endpoints that are close to the junction
    # We compute the distance between the endpoints and the junctions
    distances = np.linalg.norm(
        np.expand_dims(endpoints, axis=1) - np.expand_dims(junctions_fiber, axis=0),
        axis=2,
    )
    # We only keep the endpoints that are close to the junctions
    distances = distances < 5
    endpoints = endpoints[distances.any(axis=1)]

    retval, branches, branches_stats, _ = cv2.connectedComponentsWithStatsWithAlgorithm(
        fiber, connectivity=8, ccltype=cv2.CCL_BOLELLI, ltype=cv2.CV_16U
    )
    branches_bboxes = branches_stats[
        :,
        [cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT],
    ]

    num_branches = branches_bboxes.shape[0] - 1
    # We associate the endpoints to the branches
    endpoints_ids = np.zeros((len(endpoints),), dtype=np.uint16)
    endpoints_color = np.zeros((len(endpoints),), dtype=np.uint8)
    for i, endpoint in enumerate(endpoints):
        # Get the branch id
        branch_id = branches[endpoint[0], endpoint[1]]
        # Check if the branch id is not 0
        if branch_id != 0:
            endpoints_ids[i] = branch_id
            endpoints_color[i] = fiber[endpoint[0], endpoint[1]]

    # We remove the small branches
    kept_branches = set()
    for i in range(1, num_branches + 1):
        # Get the branch
        branch = branches == i
        # Compute the area of the branch
        area = np.sum(branch.astype(np.uint8))
        # If the area is less than 10 pixels, remove the branch
        if area < MIN_BRANCH_LENGTH:
            branches[branch] = 0
        else:
            kept_branches.add(i)

    # We remove the endpoints that are in the filtered branches
    remaining_idxs = np.isin(endpoints_ids, np.asarray(list(kept_branches)))
    if remaining_idxs.sum() == 0:
        return []
    endpoints = endpoints[remaining_idxs]

    endpoints_color = endpoints_color[remaining_idxs]
    endpoints_ids = endpoints_ids[remaining_idxs]

    # We compute the angles of the endpoints
    angles = compute_points_angle(fiber, endpoints, steps=15)
    angles = np.rad2deg(angles)
    # We compute the difference of angles between all the endpoints
    endpoints_angles_diff = cdist(angles[:, None], angles[:, None], metric="cityblock")

    # Put inf to the diagonal
    endpoints_angles_diff[range(len(endpoints)), range(len(endpoints))] = np.inf
    endpoints_distances = cdist(endpoints, endpoints, metric="euclidean")

    endpoints_distances[range(len(endpoints)), range(len(endpoints))] = np.inf

    # We sort by the distance
    endpoints_distances[endpoints_distances > MIN_BRANCH_DISTANCE] = np.inf
    endpoints_distances[endpoints_angles_diff > MIN_ANGLE] = np.inf

    matchB = np.argmin(endpoints_distances, axis=1)
    values = np.take_along_axis(endpoints_distances, matchB[:, None], axis=1)

    added_edges = dict()
    N = len(endpoints)
    A = np.eye(N, dtype=np.uint8)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if endpoints_ids[i] == endpoints_ids[j]:
                A[i, j] = 1
                A[j, i] = 1

            if matchB[i] == j and values[i, 0] < np.inf:
                added_edges[i] = j
                A[i, j] = 1
                A[j, i] = 1

    A = csr_array(A)
    n, ccs = connected_components(A, directed=False, return_labels=True)
    unique_clusters = np.unique(ccs)
    results = []
    for c in unique_clusters:
        idx = np.where(ccs == c)[0]
        branches_ids = np.unique(endpoints_ids[idx])

        unique_branches = np.logical_or.reduce(
            [branches == i for i in branches_ids], axis=0
        )

        commons_bboxes = branches_bboxes[branches_ids]
        # Compute the union of the bboxes
        min_x = np.min(commons_bboxes[:, 0])
        min_y = np.min(commons_bboxes[:, 1])
        max_x = np.max(commons_bboxes[:, 0] + commons_bboxes[:, 2])
        max_y = np.max(commons_bboxes[:, 1] + commons_bboxes[:, 3])

        new_fiber = fiber[min_y:max_y, min_x:max_x]
        new_fiber = unique_branches[min_y:max_y, min_x:max_x] * new_fiber
        for cidx in idx:
            if cidx not in added_edges:
                continue
            pointA = endpoints[cidx]
            pointB = endpoints[added_edges[cidx]]
            pointA = (
                pointA[1] - min_x,
                pointA[0] - min_y,
            )
            pointB = (
                pointB[1] - min_x,
                pointB[0] - min_y,
            )
            colA = endpoints_color[cidx]
            colB = endpoints_color[added_edges[cidx]]
            new_fiber = cv2.line(
                new_fiber,
                pointA,
                pointB,
                color=2 if colA != colB else int(colA),
                thickness=1,
            )
        # We express the bbox in the original image
        bbox = (
            coordinates[0] + min_x,
            coordinates[1] + min_y,
            max_x - min_x,
            max_y - min_y,
        )
        bbox = Bbox(x=bbox[0], y=bbox[1], width=bbox[2], height=bbox[3])
        result = Fiber(bbox=bbox, data=new_fiber)
        results.append(result)
    return results


def handle_ccs_with_junctions(
    ccs: List[np.ndarray],
    junctions: List[List[Tuple[int, int]]],
    coordinates: List[Tuple[int, int]],
):
    """
    Handle the connected components with junctions.
    The function takes a list of connected components, a list of list of junctions and a list of coordinates.
    The junctions
    The coordinates corresponds to the top left corner of the connected component.
    """
    jncts_fibers = []
    for fiber, junction, coordinate in zip(ccs, junctions, coordinates):
        jncts_fibers += handle_multiple_fiber_in_cc(fiber, junction, coordinate)

    return jncts_fibers


def extract_fibers(
    skeleton,
    skeleton_gt,
    post_process,
    x_offset: int = 0,
    y_offset: int = 0,
):
    retval, labels, stats, centroids = cv2.connectedComponentsWithStatsWithAlgorithm(
        skeleton, connectivity=8, ccltype=cv2.CCL_BOLELLI, ltype=cv2.CV_16U
    )

    bboxes = stats[
        :,
        [
            cv2.CC_STAT_LEFT,
            cv2.CC_STAT_TOP,
            cv2.CC_STAT_WIDTH,
            cv2.CC_STAT_HEIGHT,
        ],
    ]

    local_fibers = []
    coordinates = []
    junctions = []
    for i in range(1, retval):
        bbox = bboxes[i]
        x1, y1, w, h = bbox
        local_gt = skeleton_gt[y1 : y1 + h, x1 : x1 + w]
        local_label = (labels[y1 : y1 + h, x1 : x1 + w] == i).astype(np.uint8)
        local_fiber = local_gt * local_label
        local_fibers.append(local_fiber)
        coordinates.append(np.asarray([x1, y1, w, h]))
        local_junctions = find_line_intersection(local_fiber > 0)
        local_junctions = np.where(local_junctions)
        local_junctions = np.array(local_junctions).transpose()
        junctions.append(local_junctions)

    fibers = []
    if post_process:
        has_junctions = [len(j) > 0 for j in junctions]
        for fiber, coordinate in zip(
            compress(local_fibers, np.logical_not(has_junctions)),
            compress(coordinates, np.logical_not(has_junctions)),
        ):
            bbox = Bbox(
                x=coordinate[0],
                y=coordinate[1],
                width=coordinate[2],
                height=coordinate[3],
            )
            fibers.append(Fiber(bbox=bbox, data=fiber))
        # Handle fibers with junctions
        try:
            fibers += handle_ccs_with_junctions(
                compress(local_fibers, has_junctions),
                compress(junctions, has_junctions),
                compress(coordinates, has_junctions),
            )
        except (IndexError, ValueError):
            # If there is an IndexError, it means that there are no fibers with junctions
            pass
    else:
        for fiber, coordinate in zip(local_fibers, coordinates):
            bbox = Bbox(
                x=coordinate[0],
                y=coordinate[1],
                width=coordinate[2],
                height=coordinate[3],
            )
            fibers.append(Fiber(bbox=bbox, data=fiber))

    fiberprops = [FiberProps(fiber=f, fiber_id=i) for i, f in enumerate(fibers)]

    for fiber in fiberprops:
        fiber.fiber.bbox.x += x_offset
        fiber.fiber.bbox.y += y_offset

    return fiberprops


def refine_segmentation(
    image, segmentation, x_offset=0, y_offset=0, correction_model=None, device=None,
    verbose=False
):
    skeleton = skeletonize(segmentation > 0, method="zhang").astype(np.uint8)
    skeleton_gt = skeleton * segmentation

    fibers = extract_fibers(
        skeleton,
        skeleton_gt,
        post_process=True,
        x_offset=x_offset,
        y_offset=y_offset,
    )
    if correction_model is not None:
        fibers = correct_fibers(
            fibers, image, correction_model=correction_model, device=device, verbose=verbose
        )

    fibers = Fibers(fibers=fibers)
    return fibers

from numba import cuda
import numpy as np
from skimage.morphology import skeletonize, dilation
from skimage.measure import label
import math
import cv2

validIntersection = [
    [0, 1, 0, 1, 0, 0, 1, 0],
    [0, 0, 1, 0, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 1, 0, 0],
    [0, 1, 0, 0, 1, 0, 1, 0],
    [0, 0, 1, 0, 0, 1, 0, 1],
    [1, 0, 0, 1, 0, 0, 1, 0],
    [0, 1, 0, 0, 1, 0, 0, 1],
    [1, 0, 1, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 0, 1, 0, 1],
    [0, 1, 0, 1, 0, 0, 0, 1],
    [0, 1, 0, 1, 0, 1, 0, 0],
    [0, 0, 0, 1, 0, 1, 0, 1],
    [1, 0, 1, 0, 0, 0, 1, 0],
    [1, 0, 1, 0, 1, 0, 0, 0],
    [0, 0, 1, 0, 1, 0, 1, 0],
    [1, 0, 0, 0, 1, 0, 1, 0],
    [1, 0, 0, 1, 1, 1, 0, 0],
    [0, 0, 1, 0, 0, 1, 1, 1],
    [1, 1, 0, 0, 1, 0, 0, 1],
    [0, 1, 1, 1, 0, 0, 1, 0],
    [1, 0, 1, 1, 0, 0, 1, 0],
    [1, 0, 1, 0, 0, 1, 1, 0],
    [1, 0, 1, 1, 0, 1, 1, 0],
    [0, 1, 1, 0, 1, 0, 1, 1],
    [1, 1, 0, 1, 1, 0, 1, 0],
    [1, 1, 0, 0, 1, 0, 1, 0],
    [0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 1, 0, 1, 0, 1, 1],
    [1, 0, 0, 1, 1, 0, 1, 0],
    [1, 0, 1, 0, 1, 1, 0, 1],
    [1, 0, 1, 0, 1, 1, 0, 0],
    [1, 0, 1, 0, 1, 0, 0, 1],
    [0, 1, 0, 0, 1, 0, 1, 1],
    [0, 1, 1, 0, 1, 0, 0, 1],
    [1, 1, 0, 1, 0, 0, 1, 0],
    [0, 1, 0, 1, 1, 0, 1, 0],
    [0, 0, 1, 0, 1, 1, 0, 1],
    [1, 0, 1, 0, 0, 1, 0, 1],
    [1, 0, 0, 1, 0, 1, 1, 0],
    [1, 0, 1, 1, 0, 1, 0, 0],
]


@cuda.jit
def extract_intersection_keypoints_cuda(
    skeleton, intersection_points, height, width, out
):
    i, j = cuda.grid(2)

    if i > height or (j > width) or (i == 0) or (j == 0):
        return

    if skeleton[i, j] == 0:
        return
    # n_neighbors = 0
    # for ni in range(i - 1, i + 2):
    #     for nj in range(j - 1, j + 2):
    #         if ni == i and nj == j:
    #             continue
    #         n_neighbors += skeleton[ni, nj]
    # out[i, j] = n_neighbors >= 4

    # Manually extract the 3x3 neighborhood without slicing or ravel()
    neighborhood = cuda.local.array(8, dtype=np.uint8)
    idx = 0
    for ni in range(i - 1, i + 2):
        for nj in range(j - 1, j + 2):
            if ni == i and nj == j:
                continue
            neighborhood[idx] = skeleton[ni, nj]
            idx += 1
    cuda.syncthreads()
    # We check if the pixel is a valid intersection point
    for k in range(12):
        all_equal = True
        for m in range(8):
            p = neighborhood[m]
            all_equal = all_equal and p == intersection_points[k][m]
        if all_equal:
            out[i, j] = k + 1
            break


@cuda.jit
def _find_branch_angle(skeleton, endpoint, height, width, out, walk_distance=25):
    i, j = cuda.grid(2)
    if not (0 < i < height and 0 < j < width):
        return

    if not endpoint[i, j]:
        return

    # Direction offsets for 8 neighbors (clockwise from top)
    di = cuda.local.array(8, dtype=np.int8)
    dj = cuda.local.array(8, dtype=np.int8)

    di[0] = -1
    dj[0] = 0  # Top
    di[1] = -1
    dj[1] = 1  # Top-right
    di[2] = 0
    dj[2] = 1  # Right
    di[3] = 1
    dj[3] = 1  # Bottom-right
    di[4] = 1
    dj[4] = 0  # Bottom
    di[5] = 1
    dj[5] = -1  # Bottom-left
    di[6] = 0
    dj[6] = -1  # Left
    di[7] = -1
    dj[7] = -1  # Top-left

    # Current position
    curr_i = i
    curr_j = j

    steps_taken = 0
    total_angle = 0.0
    prev_dir = -1  # No previous direction

    while steps_taken < walk_distance:
        # Find next position in the skeleton
        next_i = -1
        next_j = -1
        for d in range(8):
            # Avoid backtracking to the previous pixel
            if prev_dir != -1 and (d + 4) % 8 == prev_dir:
                continue

            ni = curr_i + di[d]
            nj = curr_j + dj[d]

            # Check bounds
            if ni < 0 or ni >= height or nj < 0 or nj >= width:
                continue

            # Check if this neighbor is part of the skeleton
            if skeleton[ni, nj] == 1:
                next_i = ni
                next_j = nj
                prev_dir = d  # Remember which direction we went
                break

        # If no more skeleton pixels found, end tracing
        if next_i == -1:
            break

        # Compute angle between current and next pixel
        dx = float(next_j - curr_j)
        dy = float(next_i - curr_i)
        angle = math.atan(dy / dx)
        total_angle += angle

        # Move to the next position
        curr_i = next_i
        curr_j = next_j
        steps_taken += 1

        # # If we reach an intersection (more than 2 neighbors), stop
        # neighbor_count = 0
        # for d in range(8):
        #     ni = curr_i + di[d]
        #     nj = curr_j + dj[d]
        #     if ni >= 0 and ni < height and nj >= 0 and nj < width:
        #         if skeleton[ni, nj] == 1:
        #             neighbor_count += 1

        # if neighbor_count > 2:  # More than 2 neighbors = intersection
        #     break

    # Store average angle
    out[i, j] = total_angle / steps_taken if steps_taken > 0 else 0.0


@cuda.jit
def _find_endpoint(skeletons, height, width, endpointmap):
    i, j = cuda.grid(2)
    if i >= height or j >= width or i == 0 or j == 0:
        return
    if skeletons[i, j] == 0:
        return
    # Manually extract the 3x3 neighborhood without slicing or ravel()
    neighborhood = cuda.local.array((8,), dtype=np.uint8)
    idx = 0
    for ni in range(i - 1, i + 2):
        for nj in range(j - 1, j + 2):
            if ni == i and nj == j:
                continue
            neighborhood[idx] = skeletons[ni, nj]
            idx += 1
    total = 0
    for k in range(8):
        total += neighborhood[k]
    if total == 1:
        endpointmap[i, j] = True


def get_clean_skeleton_gpu(mask):
    """

    Create a clean skeleton from the mask using CUDA.
    Args:
        mask (np.ndarray): The input mask.
    """
    binary_mask = mask > 0

    skeleton = skeletonize(binary_mask, method="lee").copy()
    labels, counts = label(skeleton, return_num=True, connectivity=2)
    # Create a device array for the output
    # Define the grid and block size
    threads_per_block = (16, 16)
    blocks_per_grid_x = int(np.ceil(mask.shape[0] / threads_per_block[0]))
    blocks_per_grid_y = int(np.ceil(mask.shape[1] / threads_per_block[1]))
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    endpoints = cuda.device_array(mask.shape, dtype=np.uint8)
    intersection = cuda.device_array(mask.shape, dtype=np.uint8)
    orientation = cuda.device_array(mask.shape, dtype=np.float32)
    # Launch the kernel
    intersection_points_device = cuda.to_device(validIntersection)
    extract_intersection_keypoints_cuda[blocks_per_grid, threads_per_block](
        skeleton,
        intersection_points_device,
        skeleton.shape[0],
        skeleton.shape[1],
        intersection,
    )
    cuda.synchronize()
    intersection = intersection.copy_to_host() > 0

    indices = np.where(intersection)

    _find_endpoint[blocks_per_grid, threads_per_block](
        skeleton, skeleton.shape[0] - 1, skeleton.shape[1] - 1, endpoints
    )
    cuda.synchronize()

    endpoints = endpoints.copy_to_host()
    endpoint_bifurcated_branch = np.zeros_like(skeleton)

    branch_with_bifurcation = dict()
    for i in range(len(indices[0])):
        x, y = indices[0][i], indices[1][i]
        l = labels[x, y]
        branch_with_bifurcation[l] = labels == l
        endpoint_bifurcated_branch[branch_with_bifurcation[l]] = endpoints[
            branch_with_bifurcation[l]
        ]

    _find_branch_angle[blocks_per_grid, threads_per_block](
        skeleton,
        endpoint_bifurcated_branch,
        skeleton.shape[0],
        skeleton.shape[1],
        orientation,
        15,
    )
    cuda.synchronize()

    orientation = orientation.copy_to_host()

    for i in range(len(indices[0])):
        x, y = indices[0][i], indices[1][i]
        branch = branch_with_bifurcation[labels[x, y]]
        branch[x, y] = 0
        # Extract the bbox of the branch
        non_zeros = np.where(branch)
        min_x = np.min(non_zeros[0]) - 1
        max_x = np.max(non_zeros[0]) + 1
        min_y = np.min(non_zeros[1]) - 1
        max_y = np.max(non_zeros[1]) + 1

        patch_branch = branch[min_x:max_x, min_y:max_y]
        patch_orientation = orientation[min_x:max_x, min_y:max_y]
        # We remove the intersection point of the branch
        components, counts = label(patch_branch, return_num=True, connectivity=2)
        if counts < 2:
            continue
        angles = np.zeros(counts)
        size = np.zeros(counts)
        for c in range(1, counts + 1):
            try:
                angles[c - 1] = np.max(patch_orientation[components == c])
                size[c - 1] = np.sum(components == c)
            except ValueError:
                pass

        # We find the component with the closest angle to the largest component
        largest = np.argmax(size)

        largest_angle = angles[largest]
        angles[largest] = np.inf
        closest = np.argmin(np.abs(angles - largest_angle))

        for c in range(1, counts + 1):
            if c == largest + 1:
                continue
            if c == closest + 1:
                continue
            patch_branch[components == c] = 0

        patch_branch = dilation(patch_branch, np.ones((3, 3)))
        patch_branch = skeletonize(patch_branch)
        skeleton[min_x:max_x, min_y:max_y] = patch_branch
        skeleton[x, y] = 1

    return skeleton

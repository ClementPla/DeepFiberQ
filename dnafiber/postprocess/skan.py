# Functions to generate kernels of curve intersection
import numpy as np
import cv2
import itertools
from numba import njit, int64
from numba.typed import List
from numba.types import Tuple

# Define the element type: a tuple of two int64
tuple_type = Tuple((int64, int64))


def find_neighbours(fibers_map, point):
    """
    Find the next point in the fiber starting from the given point.
    The function returns None if the point is not in the fiber.
    """
    # Get the fiber id
    neighbors = []
    h, w = fibers_map.shape
    for i in range(-1, 2):
        for j in range(-1, 2):
            # Skip the center point
            if i == 0 and j == 0:
                continue
            # Get the next point
            nextpoint = (point[0] + i, point[1] + j)
            # Check if the next point is in the image
            if (
                nextpoint[0] < 0
                or nextpoint[0] >= h
                or nextpoint[1] < 0
                or nextpoint[1] >= w
            ):
                continue

            # Check if the next point is in the fiber
            if fibers_map[nextpoint]:
                neighbors.append(nextpoint)
    return neighbors


def compute_points_angle(fibers_map, points, steps=25):
    """
    For each endpoint, follow the fiber for a given number of steps and estimate the tangent line by
    fitting a line to the visited points. The angle of the line is returned.
    """
    points_angle = np.zeros((len(points),), dtype=np.float32)
    for i, point in enumerate(points):
        # Find the fiber it belongs to
        # Lets navigate along the fiber starting from the point during steps pixels.
        # We compute the angles at each step and return the mean angle.
        visited = trace_from_point(
            fibers_map > 0, (point[0], point[1]), max_length=steps
        )
        visited = np.array(visited)
        vx, vy, x, y = cv2.fitLine(visited[:, ::-1], cv2.DIST_L2, 0, 0.01, 0.01)
        # Compute the angle of the line
        points_angle[i] = np.arctan(vy / vx)

    return points_angle


def generate_nonadjacent_combination(input_list, take_n):
    """
    It generates combinations of m taken n at a time where there is no adjacent n.
    INPUT:
        input_list = (iterable) List of elements you want to extract the combination
        take_n =     (integer) Number of elements that you are going to take at a time in
                        each combination
    OUTPUT:
        all_comb =   (np.array) with all the combinations
    """
    all_comb = []
    for comb in itertools.combinations(input_list, take_n):
        comb = np.array(comb)
        d = np.diff(comb)
        if len(d[d == 1]) == 0 and comb[-1] - comb[0] != 7:
            all_comb.append(comb)
    return all_comb


def populate_intersection_kernel(combinations):
    """
    Maps the numbers from 0-7 into the 8 pixels surrounding the center pixel in
    a 9 x 9 matrix clockwisely i.e. up_pixel = 0, right_pixel = 2, etc. And
    generates a kernel that represents a line intersection, where the center
    pixel is occupied and 3 or 4 pixels of the border are ocuppied too.
    INPUT:
        combinations = (np.array) matrix where every row is a vector of combinations
    OUTPUT:
        kernels =      (List) list of 9 x 9 kernels/masks. each element is a mask.
    """
    n = len(combinations[0])
    template = np.array(([-1, -1, -1], [-1, 1, -1], [-1, -1, -1]), dtype="int")
    match = [(0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (1, 0), (0, 0)]
    kernels = []
    for n in combinations:
        tmp = np.copy(template)
        for m in n:
            tmp[match[m][0], match[m][1]] = 1
        kernels.append(tmp)
    return kernels


def give_intersection_kernels():
    """
    Generates all the intersection kernels in a 9x9 matrix.
    INPUT:
        None
    OUTPUT:
        kernels =      (List) list of 9 x 9 kernels/masks. each element is a mask.
    """
    input_list = np.arange(8)
    taken_n = [4, 3]
    kernels = []
    for taken in taken_n:
        comb = generate_nonadjacent_combination(input_list, taken)
        tmp_ker = populate_intersection_kernel(comb)
        kernels.extend(tmp_ker)
    return kernels


def find_line_intersection(input_image, show=0):
    """
    Applies morphologyEx with parameter HitsMiss to look for all the curve
    intersection kernels generated with give_intersection_kernels() function.
    INPUT:
        input_image =  (np.array dtype=np.uint8) binarized m x n image matrix
    OUTPUT:
        output_image = (np.array dtype=np.uint8) image where the nonzero pixels
                        are the line intersection.
    """
    input_image = input_image.astype(np.uint8)
    kernel = np.array(give_intersection_kernels())
    output_image = np.zeros(input_image.shape)
    for i in np.arange(len(kernel)):
        out = cv2.morphologyEx(
            input_image,
            cv2.MORPH_HITMISS,
            kernel[i, :, :],
            borderValue=0,
            borderType=cv2.BORDER_CONSTANT,
        )
        output_image = output_image + out

    return output_image


@njit
def get_neighbors_8(y, x, shape):
    neighbors = List.empty_list(tuple_type)
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < shape[0] and 0 <= nx < shape[1]:
                neighbors.append((ny, nx))
    return neighbors


@njit
def find_endpoints(skel):
    endpoints = List.empty_list(tuple_type)
    for y in range(skel.shape[0]):
        for x in range(skel.shape[1]):
            if skel[y, x] == 1:
                count = 0
                neighbors = get_neighbors_8(y, x, skel.shape)
                for ny, nx in neighbors:
                    if skel[ny, nx] == 1:
                        count += 1
                if count == 1:
                    endpoints.append((y, x))
    return endpoints


@njit
def trace_skeleton(skel):
    endpoints = find_endpoints(skel)
    if len(endpoints) < 1:
        return List.empty_list(tuple_type)  # Return empty list with proper type

    return trace_from_point(skel, endpoints[0], max_length=skel.sum())


@njit
def trace_from_point(skel, point, max_length=25):
    visited = np.zeros_like(skel, dtype=np.uint8)
    path = List.empty_list(tuple_type)

    # Check if the starting point is on the skeleton
    y, x = point
    if y < 0 or y >= skel.shape[0] or x < 0 or x >= skel.shape[1] or skel[y, x] != 1:
        return path

    stack = List.empty_list(tuple_type)
    stack.append(point)

    while len(stack) > 0 and len(path) < max_length:
        y, x = stack.pop()
        if visited[y, x]:
            continue
        visited[y, x] = 1
        path.append((y, x))
        neighbors = get_neighbors_8(y, x, skel.shape)
        for ny, nx in neighbors:
            if skel[ny, nx] == 1 and not visited[ny, nx]:
                stack.append((ny, nx))
    return path

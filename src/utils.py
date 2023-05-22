import xmltodict
from pprint import pprint
import json
from matplotlib import pyplot as plt
import numpy as np
import skimage
from typing import List, Tuple, Set
import scipy
import math


class ControlPoint:
    x: int
    y: int
    score: float
    lines: Set['ControlPointLine']

    def __init__(self, x: int, y: int, score: float) -> None:
        self.x = x
        self.y = y
        self.score = score
        self.lines = set()

    def __str__(self) -> str:
        return f'[{self.x}, {self.y}]: ({self.score})'


class ControlPointLine:
    p1: ControlPoint
    p2: ControlPoint
    score: float
    map: np.ndarray
    angle: float
    length: float

    def __init__(
        self, p1: ControlPoint, p2: ControlPoint, score: float, map: np.ndarray
    ) -> None:
        self.p1 = p1
        self.p2 = p2
        self.score = score
        self.map = map
        self.angle = math.atan2((p1.y - p2.y), (p1.x - p2.x)) / np.pi
        self.length = math.dist([p1.x, p1.y], [p2.x, p2.y])
        if self.angle < 0:
            self.angle += 1.0

    def __str__(self) -> str:
        return f'[{self.p1.x}, {self.p1.y}] -- [{self.p2.x}, {self.p2.y}]\t({self.score})'

    def has_instersection(self, other: 'ControlPointLine') -> bool:
        a1 = self.p1.x
        a2 = self.p1.y
        b1 = self.p2.x
        b2 = self.p2.y
        c1 = other.p1.x
        c2 = other.p1.y
        d1 = other.p2.x
        d2 = other.p2.y
        n = -c2 * d1 + a2 * (-c1 + d1) + a1 * (c2 - d2) + c1 * d2
        m = b2 * (c1 - d1) + a2 * (-c1 + d1) + (a1 - b1) * (c2 - d2)
        val = n / m
        return val >= 0 and val <= 1


def get_perpendicular_point(
    x1: int, y1: int, x2: int, y2: int, p1: int, p2: int
) -> Tuple[int, int, bool]:
    direction_vec = [x1 - x2, y1 - y2]
    u1 = direction_vec[0]
    u2 = direction_vec[1]
    perpendicular_vec = [direction_vec[1], -direction_vec[0]]
    v1 = perpendicular_vec[0]
    v2 = perpendicular_vec[1]
    if u1 == 0:
        print(f"p1: {p1}")
        print(f"p2: {p2}")
        print(f"[x1, y1]: [{x1}, {y1}]")
        print(f"[x2, y2]: [{x2}, {y2}]")
        out_of_bound = p2 < min(y1, y2) or p2 > max(y1, y2)
        return int(x1), int(p2), out_of_bound
    t = (p2 * u1 - p1 * u2 + u2 * x1 - u1 * y1) / (u2 * v1 - u1 * v2)
    s = (p1 + v1 * t - x1) / u1
    T = [x1 + u1 * s, y1 + u2 * s]
    out_of_bounds = False
    if -s < 0 or -s > 1:
        out_of_bounds = True
    return int(T[0]), int(T[1]), out_of_bounds


def explore_rgb_channels(
    ids: list[int], imgs_annotated: list[dict], imgs_path: str
) -> None:
    """
    Explore RGB channels of images.

    Each image is shown separately (one at a time) in popup window and after closing it the next one is shown.


    Parameters:
    -----------
    ids: list[int]
        list of IDs of images that will be shown

    imgs_annotated: list[dict]
        list of dictionaries that represent the images (from annotated data)

    imgs_path: str
        path to the folder where the actual images are stored
    """
    channels = ['Red', 'Green', 'Blue']
    for i in ids:
        img_ID = i
        img_fname = imgs_annotated[img_ID]['@name']
        img = skimage.io.imread(imgs_path + img_fname)

        fig = plt.figure()
        for c in range(3):
            plt.subplot(3, 1, c + 1)
            plt.imshow(img[:, :, c], cmap='gray')
            plt.title(channels[c] + ' channel')
        fig.suptitle(f'Image: {img_fname} \n (ID {img_ID})')
        plt.show()


def explore_blobs(
    ids: list[int], imgs_annotated: list[dict], imgs_path: str
) -> None:
    """
    Explore blobs detected in images.

    Each image is shown separately (one at a time) in popup window and after closing it the next one is shown.


    Parameters:
    -----------
    ids: list[int]
        list of IDs of images that will be shown

    imgs_annotated: list[dict]
        list of dictionaries that represent the images (from annotated data)

    imgs_path: str
        path to the folder where the actual images are stored
    """
    for i in ids:
        img_ID = i
        img_fname = imgs_annotated[img_ID]['@name']
        img = skimage.io.imread(imgs_path + img_fname)

        fig = plt.figure()

        plt.subplot(3, 1, 1)
        plt.imshow(img)
        plt.title('original')

        hue = skimage.color.rgb2hsv(img)[:, :, 0]   # the bloby one

        plt.subplot(3, 1, 2)
        plt.imshow(hue)
        plt.title('hue')

        blob = extract_blob_area(hue)

        plt.subplot(3, 1, 3)
        # plt.imshow(blob)
        plt.imshow(img, cmap='gray')
        plt.imshow(blob, alpha=0.4 * blob)
        plt.title('blob')

        fig.suptitle(f'Image: {img_fname} \n (ID {img_ID})')
        plt.show()


def explore_hsv_channels(
    ids: List[int], imgs_annotated: list[dict], imgs_path: str
) -> None:
    """
    Explore HSV channels of images.

    Each image is shown separately (one at a time) in popup window and after closing it the next one is shown.


    Parameters:
    -----------
    ids: list[int]
        list of IDs of images that will be shown

    imgs_annotated: list[dict]
        list of dictionaries that represent the images (from annotated data)

    imgs_path: str
        path to the folder where the actual images are stored
    """
    channels = ['Hue', 'Saturation', 'Value']
    for i in ids:
        img_ID = i
        img_fname = imgs_annotated[img_ID]['@name']
        img = skimage.io.imread(imgs_path + img_fname)
        gray = skimage.color.rgb2gray(img)

        fig = plt.figure()
        plt.subplot(3, 2, 1)
        plt.imshow(img)
        plt.title('original')

        img = skimage.color.rgb2hsv(img)
        plt.subplot(3, 2, 3)
        plt.imshow(img[:, :, 0])
        plt.title('hue')

        plt.subplot(3, 2, 4)
        plt.imshow(1 - img[:, :, 1])
        plt.title('saturation')

        plt.subplot(3, 2, 5)
        plt.imshow(img[:, :, 2])
        plt.title('value')

        plt.subplot(3, 2, 6)
        plt.imshow(gray)
        plt.title('gray')

        fig.suptitle(f'Image: {img_fname} \n (ID {img_ID})')
        plt.show()


def explore_thresholding(
    ids: list[int], imgs_annotated: list[dict], imgs_path: str, percentage: int
) -> None:
    """
    Explore results of thresholding.

    Each image is shown separately (one at a time) in popup window and after closing it the next one is shown.
    The thresholded map is overlayed on top of the image with color according to used color channel.
    The threshold value is computed as bottom  percentile of the whole image (based on the given percentage).


    Parameters:
    -----------
    ids: list[int]
        List of IDs of images that will be shown.

    imgs_annotated: list[dict]
        List of dictionaries that represent the images (from annotated data).

    imgs_path: str
        Path to the folder where the actual images are stored.

    percentage: int
        Bottom percentile that will be used to compute threshold value.
        If the value is 10, the thresholding will catch the 10 bottom percent of values.
    """
    for img_ID in ids:
        img_fname = imgs_annotated[img_ID]['@name']
        img = skimage.io.imread(imgs_path + img_fname)
        fig = plt.figure()
        fig.suptitle(f'Image: {img_fname} \n (ID {img_ID})')
        channels = ['Red', 'Green', 'Blue']
        for c in range(3):
            imgc = img[:, :, c]
            threshold = np.percentile(imgc, percentage)
            mask = 1.0 * (imgc < threshold)
            plt.subplot(3, 1, c + 1)
            plt.imshow(imgc, cmap='gray')
            plt.imshow(mask, alpha=mask, cmap=channels[c] + 's')
        plt.show()


def remove_border_areas(img: np.ndarray) -> np.ndarray:
    """
    Removes areas that are touching the border of the image.
    Returns a copy of the input image without the border-touching areas.

    Parameters:
    -----------
    img: np.ndarray
        binary image that will have its border-touching areas deleted
    """
    img[:, 0] = 1.0
    img[:, -1] = 1.0
    img[0, :] = 1.0
    img[-1, :] = 1.0
    img = skimage.segmentation.flood_fill(
        img, (0, 0), 0, footprint=[[0, 1, 0], [1, 1, 1], [0, 1, 0]]
    )
    return img


def extract_blob_area(img: np.ndarray) -> np.ndarray:
    """
    Extracts a blob-like area from hue channel of image in HSV colorspace that should indicate where the scar is.

    Parameters:
    -----------
    hue_channel: np.ndarray
        Hue channel of HSV colorspace image (2D array) that will have its border-touching areas deleted
    """
    # thresholding
    hue_channel = skimage.color.rgb2hsv(img)[:, :, 0]   # the bloby one
    sat_channel = skimage.color.rgb2hsv(img)[:, :, 1]   # the sharp one
    hue = 1 * (hue_channel > 0.5)

    # label the areas
    l1 = skimage.morphology.binary_closing(
        hue, footprint=skimage.morphology.disk(8)
    )
    l1 = skimage.morphology.binary_erosion(
        l1, footprint=skimage.morphology.square(3)
    )
    l1 = skimage.morphology.label(l1, connectivity=1)

    # remove small objects and re-label
    removed = skimage.morphology.remove_small_objects(l1, 70)
    l2 = skimage.morphology.label(removed, connectivity=1)

    # remove border areas
    l3, c3 = skimage.morphology.label(l2, return_num=True, connectivity=1)
    if c3 > 1:
        l3 = remove_border_areas(1 * (l3 > 0))

    # do some morphology magic to smooth and close the remaining areas
    # 20 was found experimentally as a good value for most images
    kernel_size = int(min(l3.shape) / 20)
    kernel = skimage.morphology.disk(kernel_size)
    mask = skimage.morphology.binary_closing(l3, kernel)
    return mask


def get_control_points(
    img: np.ndarray, blur_intensity: float, point_min_distance: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hue = skimage.color.rgb2hsv(img)[:, :, 0]   # the bloby one
    sat = skimage.color.rgb2hsv(img)[:, :, 1]   # the sharp one
    blob = extract_blob_area(img)
    masked = sat * blob
    masked_contrast = blob * skimage.exposure.equalize_adapthist(sat)
    blurred = skimage.filters.gaussian(masked, sigma=blur_intensity)
    return (
        skimage.feature.peak_local_max(
            blurred, min_distance=point_min_distance, exclude_border=2
        ),
        blurred,
        masked,
        masked_contrast,
    )


def compute_score(
    img: np.ndarray, kernel: np.ndarray, x: int, y: int
) -> float:
    # TODO: check if the coordinates are too close to border and the kernel would oferflow
    k = kernel.shape[0]
    xmin = x - (k // 2)
    xmax = x + (k // 2)
    ymin = y - (k // 2)
    ymax = y + (k // 2)
    chunk = img[xmin:xmax, ymin:ymax]
    score = np.sum(scipy.ndimage.convolve(chunk, kernel))
    return score


def get_gauss_kernel(radius: int) -> np.ndarray:
    kernel = np.zeros((radius, radius))
    kernel[radius // 2, radius // 2] = 1
    scipy.ndimage.gaussian_filter(kernel, radius / 4, output=kernel)
    kernel /= kernel[radius // 2, radius // 2]
    return kernel

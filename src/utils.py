import xmltodict
from pprint import pprint
import json
from matplotlib import pyplot as plt
import numpy as np
import skimage
from typing import List

def explore_rgb_channels(ids: list[int], imgs_annotated: list[dict], imgs_path: str) -> None:
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
    channels = ["Red", "Green", "Blue"]
    for i in ids:
        img_ID = i 
        img_fname = imgs_annotated[img_ID]["@name"]
        img = skimage.io.imread(imgs_path + img_fname)

        fig = plt.figure()
        for c in range(3):
            plt.subplot(3, 1, c+1)
            plt.imshow(img[:, :, c], cmap='gray')
            plt.title(channels[c]+" channel")
        fig.suptitle(f"Image: {img_fname} \n (ID {img_ID})")
        plt.show()

def explore_hsv_channels(ids: List[int], imgs_annotated: list[dict], imgs_path: str) -> None:
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
    channels = ["Hue", "Saturation", "Value"]
    for i in ids:
        img_ID = i 
        img_fname = imgs_annotated[img_ID]["@name"]
        img = skimage.io.imread(imgs_path + img_fname)
        gray = skimage.color.rgb2gray(img)

        fig = plt.figure()
        plt.subplot(3, 2, 1)
        plt.imshow(img)
        plt.title("original")

        img = skimage.color.rgb2hsv(img)
        plt.subplot(3, 2, 3)
        plt.imshow(img[:, :, 0])
        plt.title("hue")

        plt.subplot(3, 2, 4)
        plt.imshow(1-img[:, :, 1])
        plt.title("saturation")

        plt.subplot(3, 2, 5)
        plt.imshow(img[:, :, 2])
        plt.title("value")

        plt.subplot(3, 2, 6)
        plt.imshow(gray)
        plt.title("gray")

        fig.suptitle(f"Image: {img_fname} \n (ID {img_ID})")
        plt.show()


def explore_thresholding(ids: list[int], imgs_annotated: list[dict], imgs_path: str, percentage: int) -> None:
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
        img_fname = imgs_annotated[img_ID]["@name"]
        img = skimage.io.imread(imgs_path + img_fname)
        fig = plt.figure()
        fig.suptitle(f"Image: {img_fname} \n (ID {img_ID})")
        channels = ["Red", "Green", "Blue"]
        for c in range(3):
            imgc = img[:, :, c]
            threshold = np.percentile(imgc, percentage)
            mask = 1.0 * (imgc < threshold)
            plt.subplot(3, 1, c+1)
            plt.imshow(imgc, cmap='gray')
            plt.imshow(mask, alpha = mask, cmap=channels[c]+"s")
        plt.show()

def remove_border_areas(img: np.ndarray) -> np.ndarray:
    img[:, 0] = 1.0
    img[:, -1] = 1.0
    img[0, :] = 1.0
    img[-1, :] = 1.0
    img = skimage.segmentation.flood_fill(img, (0, 0), 0, footprint=[[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    return img

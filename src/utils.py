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

def explore_blobs(ids: list[int], imgs_annotated: list[dict], imgs_path: str) -> None:
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
    channels = ["Red", "Green", "Blue"]
    for i in ids:
        img_ID = i 
        img_fname = imgs_annotated[img_ID]["@name"]
        img = skimage.io.imread(imgs_path + img_fname)

        fig = plt.figure()

        plt.subplot(3, 1, 1)
        plt.imshow(img)
        plt.title('original')

        img = skimage.color.rgb2hsv(img)

        hue = img[:, :, 0]   # the bloby one

        plt.subplot(3, 1, 2)
        plt.imshow(hue)
        plt.title('hue')

        blob = extract_blob_area(hue)

        plt.subplot(3, 1, 3)
        plt.imshow(blob)
        plt.title('blob')

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
    img = skimage.segmentation.flood_fill(img, (0, 0), 0, footprint=[[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    return img

def extract_blob_area(hue_channel: np.ndarray) -> np.ndarray:
    """
    Extracts a blob-like area from hue channel of image in HSV colorspace that should indicate where the scar is. 
    
    Parameters:
    -----------
    hue_channel: np.ndarray
        Hue channel of HSV colorspace image (2D array) that will have its border-touching areas deleted
    """
    # thresholding 
    hue = 1 * (hue_channel > 0.5) 

    # label the areas
    l1 = skimage.morphology.label(hue, connectivity=1)

    # remove small objects and re-label
    removed = skimage.morphology.remove_small_objects(l1, 70)
    l2 = skimage.morphology.label(removed, connectivity=1)

    # remove border areas
    l3, c3 = skimage.morphology.label(l2, return_num=True, connectivity=1)
    if c3 > 1:
        l3 = remove_border_areas(1 * (l3 > 0))

    # do some morphology magic to smooth and close the remaining areas
    kernel_size = int(min(l3.shape) / 20) # 20 was found experimentally as a good value for most images
    kernel = skimage.morphology.disk(kernel_size)
    return skimage.morphology.binary_closing(l3, kernel)

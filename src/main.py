import xmltodict
from pprint import pprint
import json
from matplotlib import pyplot as plt
import numpy as np
from utils import *
import skimage
import typing

ANNOTATION_PATH = "./data/annotations.xml"
IMAGES_PATH = "./data/images/default/"

with open(ANNOTATION_PATH) as f:
    doc = xmltodict.parse(f.read())

imgs_annotated = doc["annotations"]["image"]

def explore_channels(ids: list[int], imgs_annotated: list[dict], imgs_path: str) -> None:
    """ 
    Explore collor channels of images.

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

explore_channels(list(range(20)), imgs_annotated, IMAGES_PATH)

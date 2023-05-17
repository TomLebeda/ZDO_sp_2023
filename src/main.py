import xmltodict
from pprint import pprint
import json
from matplotlib import pyplot as plt
import numpy as np
from utils import *
import skimage
import typing

ANNOTATION_PATH = './data/annotations.xml'
IMAGES_PATH = './data/images/default/'
GOOD_IMAGES_ID = [4, 14, 17, 18, 20, 24, 28, 33, 38, 44, 45, 52, 56, 57, 88]

with open(ANNOTATION_PATH) as f:
    doc = xmltodict.parse(f.read())

imgs_annotated = doc['annotations']['image']

# explore_rgb_channels(list(range(50)), imgs_annotated, IMAGES_PATH)
explore_hsv_channels(list(range(100)), imgs_annotated, IMAGES_PATH)
# explore_thresholding(GOOD_IMAGES_ID, imgs_annotated, IMAGES_PATH, 10)


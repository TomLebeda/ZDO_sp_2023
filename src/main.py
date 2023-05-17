import xmltodict
from pprint import pprint
import json
from matplotlib import pyplot as plt
import numpy as np
from utils import *
import skimage.io
import typing

ANNOTATION_PATH = './data/annotations.xml'
IMAGES_PATH = './data/images/default/'
GOOD_IMAGES_ID = [14, 17, 18, 20, 24, 28, 33, 38, 44, 45, 52, 56, 57]

with open(ANNOTATION_PATH) as f:
    doc = xmltodict.parse(f.read())

imgs_annotated = doc['annotations']['image']

# explore_rgb_channels(list(range(50)), imgs_annotated, IMAGES_PATH)
# explore_hsv_channels(list(range(100)), imgs_annotated, IMAGES_PATH)
# explore_thresholding(GOOD_IMAGES_ID, imgs_annotated, IMAGES_PATH, 10)


for img_ID in list(range(100)):
    img_fname = imgs_annotated[img_ID]['@name']
    img = skimage.io.imread(IMAGES_PATH + img_fname)

    plt.subplot(4, 1, 1)
    plt.imshow(img)
    plt.title('original')

    img = skimage.color.rgb2hsv(img)

    hue = img[:, :, 0]   # the bloby one
    sat = img[:, :, 1]   # the sharp one

    plt.subplot(4, 1, 2)
    plt.imshow(hue)
    plt.title('hue')

    hue = 1.0 * (hue > 0.5)

    plt.subplot(4, 1, 3)
    plt.imshow(hue)
    plt.title('thresholded')

    labeled = skimage.morphology.label(hue)
    # print(labeled.size)
    # if labeled.size > 1:
    hue = remove_border_areas(hue)
    kernel_size = int(min(hue.shape) / 20) # 20 was found experimentally as a good value for most images
    hue = skimage.morphology.binary_closing(hue, skimage.morphology.disk(kernel_size))
    hue = skimage.morphology.remove_small_objects(hue, 70) # 50 was found experimentally as a good value for most images
    # hue = skimage.morphology.remove_small_holes(hue, 64) # 50 was found experimentally as a good value for most images

    plt.subplot(4, 1, 4)
    plt.imshow(hue)
    plt.title('edited')

    plt.show()

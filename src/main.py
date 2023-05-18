import xmltodict
from matplotlib import pyplot as plt
from utils import *
import skimage

ANNOTATION_PATH = './data/annotations.xml'
IMAGES_PATH = './data/images/default/'
GOOD_IMAGES_ID = [14, 17, 18, 20, 24, 28, 33, 38, 44, 45, 52, 56, 57]

with open(ANNOTATION_PATH) as f:
    doc = xmltodict.parse(f.read())

imgs_annotated = doc['annotations']['image']

# explore_rgb_channels(list(range(50)), imgs_annotated, IMAGES_PATH)
explore_blobs(list(range(50)), imgs_annotated, IMAGES_PATH)
# explore_hsv_channels(list(range(100)), imgs_annotated, IMAGES_PATH)
# explore_thresholding(GOOD_IMAGES_ID, imgs_annotated, IMAGES_PATH, 10)

# for img_ID in list(range(100)):
# img_ID = 14
# img_fname = imgs_annotated[img_ID]['@name']
# img = skimage.io.imread(IMAGES_PATH + img_fname)
#
# plt.subplot(3, 1, 1)
# plt.imshow(img)
# plt.title('original')
#
# img = skimage.color.rgb2hsv(img)
#
# hue = img[:, :, 0]   # the bloby one
# sat = img[:, :, 1]   # the sharp one
#
# plt.subplot(3, 1, 2)
# plt.imshow(hue)
# plt.title('hue')
#
# blob = extract_blob_area(hue)
#
# plt.subplot(3, 1, 3)
# plt.imshow(blob)
# plt.title('blob')
#
# plt.show()

import xmltodict
from pprint import pprint
import json
from matplotlib import pyplot as plt
import numpy as np
from utils import *
from utils_Tupy import *
import skimage.io
import typing
from skimage.transform import (hough_line, hough_line_peaks,
                               probabilistic_hough_line)
from skimage import filters
from skimage.feature import canny
import skimage.transform
from skimage import measure
from skimage import io, morphology
from skimage.measure import ransac
from skimage.transform import PolynomialTransform
from scipy.stats import linregress
from statistics import mean



ANNOTATION_PATH = './data/annotations.xml'
IMAGES_PATH = './data/images/default/'
GOOD_IMAGES_ID = [14, 17, 18, 20, 24, 28, 33, 38, 44, 45, 52, 56, 57]
LIMIT_CRIT_J = 25

with open(ANNOTATION_PATH) as f:
    doc = xmltodict.parse(f.read())

imgs_annotated = doc['annotations']['image']

# explore_rgb_channels(list(range(50)), imgs_annotated, IMAGES_PATH)
# explore_hsv_channels(list(range(100)), imgs_annotated, IMAGES_PATH)
# explore_thresholding(GOOD_IMAGES_ID, imgs_annotated, IMAGES_PATH, 10)
#
for img_ID in range(100):
# img_ID = 38
    print('*'*15, f'img_ID: {img_ID}', '*'*15)
    img_fname = imgs_annotated[img_ID]["@name"]
    img_original = skimage.io.imread(IMAGES_PATH + img_fname)
    img = skimage.color.rgb2hsv(img_original)
    img_hsv = img[:, :, 1] # hsv: value gray

    # ****************************************************************************************
    # ****                              A L G O R I T M U S                               ****
    # ****************************************************************************************

    # Prahovani - Otsu metoda
    threshold = filters.threshold_otsu(img_hsv)    # prah
    binary_image = img_hsv > threshold             # naprahovany obrazek

    # Skeletonizace a detekce prímek pomoci Houghovy transformace
    skeleton = skimage.morphology.skeletonize(binary_image)
    lines = probabilistic_hough_line(skeleton, threshold=20, line_length=5, line_gap=2) # (20,5,2)
    print(f'threshold - binary image:\t {threshold:0.2f}')

    # Morfologice operace - Opening
    # kernel_big = skimage.morphology.diamond(2)
    # binary_image = skimage.morphology.binary_opening(binary_image , kernel_big)

    # Detekce přímek pomocí Houghovy transformace
    # edges = filters.sobel(binary_image)
    # lines = probabilistic_hough_line(edges, threshold=10, line_length=180, line_gap=10)

    # from binary image
    # lines = probabilistic_hough_line(binary_image, threshold=20, line_length=5, line_gap=2) # (20,5,2)


    # ------------------ JEDNOTLIVE BODY ----------------------
    ''' Z lines vybere jednotlive body a seradi podle souradnice x.'''
    point_list = list()
    for line in lines:
        for point in line:
            if point not in point_list:                   # kontrola, zda uz tam dany bod neni
                point_list.append(point)

    sorted_list = sorted(point_list, key=lambda x: x[0])  # serazeni bodu podle souradnice

    x = list()    # inicializace listu se souradnici x
    y = list()    # inicializace listu se souradnici y

    for point in sorted_list:
        if point[0] not in x:
            x.append(point[0])
            y.append(point[1])
    # x: list - [x1, x2 , ...]
    # y: list - [y1, y2 , ...]
    # ------------------------------------------------------

    list_dist = get_list_dist(x, y)                    # linearni regrese -> vrati celkovou vzdalenost bodu od primky
    x, y = remove_distant_points(x, y, list_dist)      # odstrani moc vzdelene body od primky podle 'thresholdu'
    list_dist = get_list_dist(x, y)                    # vzdalenost po odstraneni bodů
    crit_J = get_best_2_lines_with_crit_J(x, y)        # vypocet nejlesiho rozdeleni 1line -> 2line, vrati hodnotu J


    # Vypocet kriterialni funkce
    min_J_2line = min(crit_J)                 # kriterialni hodnota pro 2 liny
    min_J_1line = sum(list_dist)              # kriterialní hodnota pro 1 linu
    diff_J = abs(min_J_1line - min_J_2line)   # rozdil mezi hodnotou J mezi 1 a dvemi linami

    # vypis do kozole
    print('-'*7, 'CRITERION', '-'*7)
    print(f'min 1line:\t {min_J_1line:7.2f}')
    print(f'min 2line:\t {min_J_2line:7.2f}')
    print(f'difference:\t {diff_J:7.2f}')
    print('-'*25)
    #---------------------

    # x, y pro 2 lines = jizva jako lomenna cara
    index_min = crit_J.index(min(crit_J))
    x_list = [x[:index_min+1], x[index_min:]]
    y_list = [y[:index_min+1], y[index_min:]]

    # TODO: result bud 1 nebo 2 čary podle crit J -> vysledek reprezentovat koncovyma bodoma


    # ****************************************************************************************
    # ****                             V Y K R E S L E N I                                ****
    # ****************************************************************************************
    # TODO: dat podminku pro vykresleni pro parametr

    # figure_polyline(img_original, lines) # good for 14 (20,10,2)
    # figure_result(img_hsv, skeleton, img_original, lines, img_ID)
    # figure_skeleton_contours(skeleton, img_original)
    # figure_one_hough_line(skeleton, img_original, img_ID)         # good for 15,14 with opening dia 2 before skeleton
    # plot_histogram(list_dist)                                     # vykresleni histogramu vzdalenosti
    figure_polyline_from_point(img_original, sorted_list)           # vykresli linu ze serazenich bodu podle x


    #--------------------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(8, 8))
    # ----- Vykresleni 2 line -----
    plt.subplot(211)
    plt.scatter(x, y, color='blue', label='Data') # Vykreslení bodů
    plot_img_with_regress_line_and_scatter(x_list, y_list, img_original, img_ID)
    plt.title(f'Input image ID: {img_ID}, line: 2, J_crit = {min_J_2line:.2f}, diff = {diff_J:.2f}')

    # ----- Vykresleni 1 line -----
    plt.subplot(212)
    plt.scatter(x, y, color='blue', label='Data') # Vykreslení bodů
    plot_img_with_regress_line_and_scatter([x], [y], img_original, img_ID)
    plt.title(f'Input image ID: {img_ID}, line: 1, J_crit = {min_J_1line:.2f}')


    # Zobrazení grafu
    plt.show()

    #--------------------------------------------------------------------------------------
    # fig, axes = plt.subplots(1, 1, figsize=(8, 8))
    #
    # # Vykreslení nalezených přímek
    # axes.imshow(img_original, cmap='gray')
    # # Řád polynomu (1 pro přímku)
    # degree = 12
    #
    # # Regrese polilinie
    # coefficients = np.polyfit(x, y, degree)
    # polynomial = np.poly1d(coefficients)
    #
    # # Generování hodnot pro vykreslení přímky
    # x_line = np.linspace(min(x), max(x), 100)
    # y_line = polynomial(x_line)
    #
    # # Vykreslení bodů polilinie
    # plt.scatter(x, y, color='blue', label='Data')
    #
    # # Vykreslení regresní přímky
    # plt.plot(x_line, y_line, color='red', label='Regression Line')
    #
    # # Popisky os
    # plt.xlabel('X')
    # plt.ylabel('Y')
    #
    # # Legenda
    # plt.legend()
    #
    # # Zobrazení grafu
    # plt.show()




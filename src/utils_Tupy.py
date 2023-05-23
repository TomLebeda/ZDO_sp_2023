from typing import Tuple

import matplotlib.pyplot as plt
import skimage
from skimage import measure
from skimage.transform import (hough_line, hough_line_peaks,
                               probabilistic_hough_line)
import numpy as np
from scipy.stats import linregress

from utils import extract_blob_area


def figure_result(img_gray, binary_image, skeleton, img_original, lines,ID):
    """ Vykresí 3 obrázky v 1 figure:
            1. obrázek v šedi
            2. obrázek po prahování
            3. obrázek skeletonizace
            4. originální obrázek s polylinama
    """
    fig, axes = plt.subplots(4, 1, figsize=(6, 10))

    # axes[0].imshow(img_gray, cmap=plt.cm.gray)
    axes[0].imshow(img_gray)
    axes[0].set_title('HSV image')

    # plt.subplot(132)
    axes[1].imshow(binary_image, cmap='gray', )
    axes[1].set_title('binary image')

    # plt.subplot(132)
    axes[2].imshow(skeleton, cmap='gray', )
    axes[2].set_title('skeleton')

    figure_polyline(img_original, lines, axes[3], False)
    fig.suptitle(f'Input image: {ID}')


def figure_polyline(img_original, lines: list, axes=None, figure=True):
    """ Vykresí originální obrázek s polylinama 1 figure.

     Parameters:
    -----------
        - img_original: originalni obrazek
        - lines: list car - koncovy a zacatecnicke souradnice ((x0,y0), (x1,y1))
        - axes: preda osu na ktere se maji liny vykreslit (pro subploty)
        - figure: vykresleni do vlastni figure = True (default), jinak False
     """

    if figure:
        fig, axes = plt.subplots(1, 1, figsize=(8, 8))

    # Vykreslení nalezených přímek
    axes.imshow(img_original, cmap='gray')
    axes.set_title('find polyline')
    plot_lines(lines, axes)

    # Zobrazení výsledků
    # plt.tight_layout()
    # plt.show()


def plot_lines(lines: list, axes=None) -> None:
    for line in lines:
        p0, p1 = line
        if axes:
            axes.plot((p0[0], p1[0]), (p0[1], p1[1]), '-r')
        else:
            plt.plot((p0[0], p1[0]), (p0[1], p1[1]), '-r')



def figure_polyline_from_point(img_original, lines: list):
    """ Vykresí originální obrázek s polylinama ze serazenych bodu podle x osy do 1 figure.

     Parameters:
    -----------
        - img_original: originalni obrazek
        - lines: list car - jednotlive body [(x0,y0), (x1,y1), ...]
     """
    fig, axes = plt.subplots(1, 1, figsize=(8, 8))

    # Vykreslení nalezených přímek
    axes.imshow(img_original, cmap='gray')
    axes.set_title('find polyline (sorted) - before remove distance points')

    #----- serazene jednotlive body----
    for i in range(len(lines)-1):
        p0 = lines[i]
        p1 = lines[i+1]

        # print(line)
        axes.plot((p0[0], p1[0]), (p0[1], p1[1]), '-r')
    # plt.show()


def figure_skeleton_contours(skeleton, img_original):
    # Find the contours from the skeleton
    contours = measure.find_contours(skeleton, level=0.5)

    # Plot the polyline on the original image
    plt.imshow(img_original, cmap='gray')
    for i in contours:
        plt.plot(i[:, 1], i[:, 0], '-r')
    # plt.show()


def figure_one_hough_line(skeleton, img_original, img_ID):
    h, theta, d = hough_line(skeleton)

    plt.figure(figsize=(15, 15))

    plt.subplot(131)
    plt.imshow(skeleton,
               cmap=plt.cm.gray
               )
    plt.title('Input image')

    plt.subplot(132)
    plt.imshow(np.log(1 + h),
               extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]),
                       d[-1], d[0]],
               #            cmap=plt.cm.gray,
               aspect=1/1.5)
    plt.title('Hough transform')
    plt.xlabel('Angles (degrees)')
    plt.ylabel('Distance (pixels)')

    gray = skimage.color.rgb2gray(img_original)

    plt.subplot(133)
    plt.imshow(img_original, cmap=plt.cm.gray)
    rows, cols = gray.shape
    for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
        y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
        y1 = (dist - cols * np.cos(angle)) / np.sin(angle)
        plt.plot((0, cols), (y0, y1), '-r')
        break
    plt.axis((0, cols, rows, 0))
    plt.title(f'Detected lines ID {img_ID}')
    plt.show()


def dist_point_from_line(x: float, y: float, slope, intercept):
    distance = np.abs(slope * x - y + intercept) / np.sqrt(slope**2 + 1)
    return distance


def get_list_dist(x: list, y: list):
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    # print(f'std error: {std_err}')
    list_dist = list()
    for i in range(len(x)):
        dist_i = dist_point_from_line(x[i], y[i], slope, intercept)
        list_dist.append(dist_i)
    return list_dist


def remove_distant_points(x, y, list_dist):
    """ Odstrani automaticky přiliš vzdálené body od přímky podle 'thresholdu'. """
    # hranični body pro treshold : THRESHOLD_MIN >= threshold >= THRESHOLD_MAX (TODO: udelat poměrově k vel. obrazku?)
    THRESHOLD_MIN = 5
    THRESHOLD_MAX = 20

    # Výpočet histogramu
    histogram, bin_edges = np.histogram(list_dist, bins='auto')

    indices = np.where(histogram == histogram.min()) # minimalni hodnota z histogramu
    idx = round(len(indices[0])/2)-1                 # vezmi "prostredni" minimálni index
    idx = max(idx, 0)                                # pokud idx zaporný -> nulty index
    idx = min(idx, len(indices)-1)
    threshold = bin_edges[indices[0][idx]]           # thhreshold pro odstraneni bodu
    threshold = min(np.float32(THRESHOLD_MAX), threshold)      # TODO: zmenit, ted nesmi byt vzdalenjesi > 20
    threshold = max(np.float32(THRESHOLD_MIN), threshold)      # TODO: zmenit, ted nesmi byt vzdalenjesi < 5
    indices = np.where(list_dist > threshold)
    print(f'threshold - remove points:\t {threshold:0.2f}')
    # print(indices, idx)
    # print(list_dist)
    # print('histogram', histogram)

    x_len = len(x)
    # Odstranění hodnot na indexech
    for index in sorted(indices[0], reverse=True):
        # print(index, type(index))
        del x[index]
        del y[index]
    print(f'Removed {len(indices[0])}/{x_len} points.')
    return x, y


def get_lines_points(x: list, y: list) -> list[list[list: int, float]]:
    ''' Vraci listu listu se startovacími/konečnými body liny.'''
    ROUND_DECIMALS = 2
    incision_polyline = list()
    for i in range(len(x)):
        try:
            slope, intercept, r_value, p_value, std_err = linregress(x[i], y[i])
        except RuntimeWarning:
            print('Skipped 2 line -> same as 1 line.')
            continue

        # urceni startovaciho bodu polyline [x_start, y_start]
        index_min = x[i].index(min(x[i]))
        x_start = x[i][index_min]
        y_start = round(intercept + slope * x_start, ROUND_DECIMALS)

        # urceni koncoveho bodu polyline [x_end, y_end]
        index_max = x[i].index(max(x[i]))
        x_end = x[i][index_max]
        y_end = round(intercept + slope * x_end, ROUND_DECIMALS)

        incision_polyline.append([[y_start, x_start], [y_end, x_end]])
    return incision_polyline


def plot_regress_line(incision_polyline):
    line_array = np.array(incision_polyline)
    plt.plot(line_array[:, 1], line_array[:, 0], color='red', label='Regression Line')


def plot_img_with_regress_line(incision_polyline, img_original, img_ID: int) -> None:
    # Vykreslení originalniho obrazku
    plt.imshow(img_original, cmap='gray')

    for i in range(len(incision_polyline)):
        plot_regress_line(incision_polyline[i]) # Vykreslení nalezených přímek
    plt.legend()
    # plt.title(f'Input image ID: {img_ID}, line: {len(x)}')


def plot_histogram(distance: list) -> None:
    # # Vykreslení histogramu a lokálních minim
    plt.hist(distance, bins='auto', edgecolor='black')
    # plt.plot(histogram, color='b')
    # plt.axvline(x=threshold, color='r', linestyle='--')
    plt.xlabel('Bin')
    plt.ylabel('Počet')
    plt.title('Histogram s lokálními minimy')
    plt.show()


def get_control_points(
        img: np.ndarray, blur_intensity: float, point_min_distance: float
) -> Tuple[np.ndarray, np.ndarray]:
    hue = skimage.color.rgb2hsv(img)[:, :, 0]   # the bloby one
    sat = skimage.color.rgb2hsv(img)[:, :, 1]   # the sharp one
    blob = extract_blob_area(hue)
    masked = sat * blob
    blurred = skimage.filters.gaussian(masked, sigma=blur_intensity)
    return (
        skimage.feature.peak_local_max(
            blurred, min_distance=point_min_distance, exclude_border=2
        ),
        blurred,
    )


def get_best_2_lines_with_crit_J(x: list, y: list) -> float:
    ''' Vypocte nejlepsi rozdeleni z 1 liny na 2 liny.'''
    crit_J = list()
    for i in range(2, len(x)-1):
        x1 = x[:i]
        x2 = x[i:]
        y1 = y[:i]
        y2 = y[i:]

        list_dist1 = get_list_dist(x1, y1)
        list_dist2 = get_list_dist(x2, y2)
        criterial = sum(list_dist1) + sum(list_dist2)
        crit_J.append(criterial)
    return crit_J


def save_figure(file_name: str) -> None:
    '''Uloží figure'''
    plt.axis('off')                         # vypne osy
    if plt.gca().get_legend() is not None:  # odstrani legendu pokud existuje
        plt.gca().get_legend().remove()
    plt.savefig(file_name, format="pdf", bbox_inches="tight", pad_inches=0)


def init_data():
    data = dict()
    data['filename'] = ''
    data['incision_polyline'] = list(list())
    data['crossing_positions'] = list()
    data['crossing_angles'] = list()
    return data




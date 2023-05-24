# Coding: utf-8

from utils import *
from utils_Tupy import *
import skimage.io
from skimage.transform import probabilistic_hough_line
from skimage import filters
import skimage.transform
from skimage import morphology
import os



ANNOTATION_PATH = './data/annotations.xml'
IMAGES_PATH = './data/images/default/'
GOOD_IMAGES_ID = [14, 17, 18, 20, 24, 28, 33, 38, 44, 45, 52, 56, 57]
LIMIT_CRIT_J = 25
SAVE = True
PRINT = True

with open(ANNOTATION_PATH) as f:
    doc = xmltodict.parse(f.read())

imgs_annotated = doc['annotations']['image']

#
for img_ID in range(100):
    # img_ID = 70
    print('\n\n', '*'*15, f'img_ID: {img_ID}', '*'*15)
    img_fname = imgs_annotated[img_ID]["@name"]
    img_original = skimage.io.imread(IMAGES_PATH + img_fname)
    img = skimage.color.rgb2hsv(img_original)
    # gray = skimage.color.rgb2hsv(img_original)
    # blob = extract_blob_area(img)
    img_hsv = (0.15*img[:, :, 0] + 0.7 * img[:, :, 0] * img[:, :, 1] + 2 * img[:, :, 1]+ -0.7*img[:, :, 2])  # hsv: value gray

    # ****************************************************************************************
    # ****                              A L G O R I T M U S                               ****
    # ****************************************************************************************

    # Prahovani - Otsuova metoda
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
    # FORMAT:
    # x: list - [x1, x2 , ...]
    # y: list - [y1, y2 , ...]
    # ------------------------------------------------------
    if len(x) < 4:
        print(f'\t- WARNING: image ID {img_ID} skipped. Few points left.')
        continue

    list_dist = get_list_dist(x, y)                    # linearni regrese -> vrati celkovou vzdalenost bodu od primky
    x, y = remove_distant_points(x, y, list_dist)      # odstrani moc vzdelene body od primky podle 'thresholdu'

    if len(x) < 4:
        print(f'\t- WARNING: image ID {img_ID} skipped. Few points left.')
        continue

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

    # result bud 1 nebo 2 čary podle crit J -> vysledek reprezentovat koncovyma bodoma
    incision_polyline2 = get_lines_points(x_list, y_list)
    incision_polyline1 = get_lines_points([x], [y])

    print(f'Incision polyline2:\t {incision_polyline2}')
    print(f'Incision polyline1:\t {incision_polyline1}')

    # vyber 1 nebo 2 line podle kriteria
    if diff_J > LIMIT_CRIT_J:
        result_incision_polyline = incision_polyline2
    else:
        result_incision_polyline = incision_polyline1


    # ****************************************************************************************
    # ****                             V Y K R E S L E N I                                ****
    # ****************************************************************************************
    if PRINT:
        # figure_polyline(img_original, lines) # good for 14 (20,10,2)
        figure_result(img_hsv, binary_image, skeleton, img_original, lines, img_ID)
        # figure_skeleton_contours(skeleton, img_original)
        # figure_one_hough_line(skeleton, img_original, img_ID)         # good for 15,14 with opening dia 2 before skeleton
        # plot_histogram(list_dist)                                     # vykresleni histogramu vzdalenosti
        # figure_polyline_from_point(img_original, sorted_list)         # vykresli linu ze serazenich bodu podle x

        # --------------------------------------------------------------------------------------
        fig, axes = plt.subplots(2, 1, figsize=(8, 8))
        fig.suptitle(f'Input image: {img_ID}\n\ndiff = {diff_J:.2f}')
        # ----- Vykresleni 2 line -----
        plt.subplot(211)
        plt.scatter(x, y, color='blue', label='Data')  # Vykreslení bodů
        plot_img_with_regress_line(incision_polyline2, img_original, img_ID)
        plt.title(f'line: 2, J_crit = {min_J_2line:.2f}')

        # ----- Vykresleni 1 line -----
        plt.subplot(212)
        plt.scatter(x, y, color='blue', label='Data')  # Vykreslení bodů
        plot_img_with_regress_line(incision_polyline1, img_original, img_ID)
        plt.title(f'line: 1, J_crit = {min_J_1line:.2f}')

        # Zobrazení grafu
        # plt.show()

        # --------------------------------------------------------------------------------------
        fig, axes = plt.subplots(2, 1, figsize=(8, 8))
        fig.suptitle(f'Input image: {img_ID}')
        # ----- Vykresleni vysledku s scatter -----
        plt.subplot(211)
        plt.scatter(x, y, color='blue', label='Data')  # Vykreslení bodů
        plot_img_with_regress_line(result_incision_polyline, img_original, img_ID)
        plt.legend().set_visible(False)
        plt.title(f'Result: incision polyline with points')

        # ----- Vykresleni vysledku bez scatter -----
        plt.subplot(212)
        plot_img_with_regress_line(result_incision_polyline, img_original, img_ID)
        plt.legend().set_visible(False)
        plt.title(f'Result: incision polyline')

        # Zobrazení grafu
        plt.show()

    #--------------------------------------------------------------------------------------
    # -----              V Y K R E S L E N I   P R O    U L O Z E N I                 -----
    #--------------------------------------------------------------------------------------


    if SAVE:
        # pokud neexistuje složka 'figure' -> vytvori se
        if not os.path.exists('figure'):
            os.mkdir('figure')

        MARKER_SIZE = 12     # SCATTER PLOT: velikost markeru
        LINE_WIDTH = 0.2     # SCATTER PLOT: sirka okraje

        # ----- Ulozeni 2 line -----
        plt.figure(frameon=False)  # figure bez ramecku
        plt.scatter(x, y, color='blue', s=MARKER_SIZE, linewidths=LINE_WIDTH, edgecolor ="black") # Vykreslení bodů
        plot_img_with_regress_line(incision_polyline2, img_original, img_ID)
        save_figure(f"figure/{img_ID:03d}_2line.pdf")

        # ----- Ulozeni 1 line -----
        plt.clf()      # nemusim vytvaret novou -> staci vymazat predchozi
        plt.scatter(x, y, color='blue', s=MARKER_SIZE, linewidths=LINE_WIDTH, edgecolor ="black") # Vykreslení bodů
        plot_img_with_regress_line(incision_polyline1, img_original, img_ID)
        save_figure(f"figure/{img_ID:03d}_1line.pdf")

        # ----- Ulozeni vysledku -----
        plt.clf()
        plot_img_with_regress_line(result_incision_polyline, img_original, img_ID)
        save_figure(f"figure/{img_ID:03d}_result.pdf")

        # ----- ulozeni vstupniho obrazku -----
        plt.clf()
        plt.imshow(img_original, cmap='gray')
        save_figure(f"figure/{img_ID:03d}_original_image.pdf")

        # ----- ulozeni obrazku v RGB2HSV -----
        plt.clf()
        plt.imshow(img_hsv)
        save_figure(f"figure/{img_ID:03d}_hsv_image.pdf")

        # ----- ulozeni skeletonu -----
        plt.clf()
        plt.imshow(skeleton, cmap='gray')
        save_figure(f"figure/{img_ID:03d}_skeleton.pdf")

        # ----- ulozeni binarniho obrazku -----
        plt.clf()
        plt.imshow(binary_image, cmap='gray')
        save_figure(f"figure/{img_ID:03d}_binary_image.pdf")

        # ----- ulozeni binarniho obrazku -----
        plt.clf()
        figure_skeleton_contours(skeleton, img_original)
        save_figure(f"figure/{img_ID:03d}_skeleton_contours.pdf")

        # ----- ulozeni Hough lines -----
        plt.clf()
        plt.imshow(img_original, cmap='gray')
        plot_lines(lines)
        save_figure(f"figure/{img_ID:03d}_hough_lines.pdf")
        plt.close('all')


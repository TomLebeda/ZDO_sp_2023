import xmltodict
from matplotlib import pyplot as plt
from utils import *
import time

ANNOTATION_PATH = '/home/tom/School/FAV/8_semestr/ZDO/ZDO_sp_2023/data/annotations.xml'
IMAGES_PATH = '/home/tom/School/FAV/8_semestr/ZDO/ZDO_sp_2023/data/images/default/'

ANNOTATION_PATH = 'data/annotations.xml'
IMAGES_PATH = 'data/images/default/'

BLUR_INTENSITY = 2   # blur intensity for control point searching
CONTROL_POINT_MIN_DIST = 1   # minimal distance for control points
POINT_SCORE_KERNEL_SIZE = (
    5  # size of the kernel used to compute score of control points
)
CONTROL_POINT_MIN_COUNT = (
    3  # minimum number of control points needed to process image
)
CONTROL_POINT_FILTER_THRESHOLD = (
    100  # if there is more than that control points detected,
    # the filtering mechanism will be triggered to reduce number of control points
)
LINE_SCORE_KERNEL_BLUR_INTENSITY = (
    1  # intensity of blur when computing brightness score for given line
)
STITCH_ANGLE_COUNT = (
    20  # how many angles to try when detecting stitch-line angles
)

with open(ANNOTATION_PATH) as f:
    doc = xmltodict.parse(f.read())

imgs_annotated = doc['annotations']['image']

# explore_rgb_channels(list(range(50)), imgs_annotated, IMAGES_PATH)
# explore_blobs(list(range(50)), imgs_annotated, IMAGES_PATH)
# explore_hsv_channels(list(range(100)), imgs_annotated, IMAGES_PATH)
# explore_thresholding(GOOD_IMAGES_ID, imgs_annotated, IMAGES_PATH, 10)


def init_data():
    data = dict()
    data['filename'] = ''
    data['incision_polyline'] = list(list())
    data['crossing_positions'] = list()
    data['crossing_angles'] = list()
    return data


def run_find_incisions(path: str, save_fig: bool, verbose: bool):
    # load the image
    # for img_ID in [1, 11, 37, 3, 41, 8]:
    for img_ID in range(1):
        # img_ID = 1
        img_fname = imgs_annotated[img_ID]['@name']
        img = skimage.io.imread(IMAGES_PATH + img_fname)
        t0 = time.time()
        if verbose:
            print(f'Loading image {path}... ')
        # img = skimage.io.imread(path)
        if img.shape[0] > img.shape[1]:
            print(
                'Warning: Vertical image detected, will rotate by 90 degrees.'
            )
            img = skimage.transform.rotate(img, 90.0, resize=True)
        if verbose:
            print(f'Done ({(time.time() - t0):.4f} s)')

        # prepare some variables
        control_points: List[ControlPoint] = list()
        control_points_score_kernel = get_gauss_kernel(POINT_SCORE_KERNEL_SIZE)

        # get the control points and associated images
        if verbose:
            t0 = time.time()
            print(f'\nExtracting control points... ')
        xy, blurred, masked, masked_contrast = get_control_points(
            img, BLUR_INTENSITY, CONTROL_POINT_MIN_DIST
        )
        if verbose:
            print(f'Done ({(time.time() - t0):.4f} s)')

        # check if enough control points have been detected
        if len(xy) < CONTROL_POINT_MIN_COUNT:
            print(
                'Warning: Not enough control points detected, image can not be processed further.'
            )
            return

        # construct control points and compute their score
        if verbose:
            t0 = time.time()
            print(f'\nProcessing control points... ')
        control_point_scores = np.zeros(len(xy))
        for i, c in enumerate(xy):
            score = compute_score(
                masked, control_points_score_kernel, c[0], c[1]
            )
            control_point_scores[i] = score
            control_points.append(ControlPoint(c[0], c[1], score))
        control_points.sort(key=lambda c: c.score, reverse=True)

        # check number of control points and trigger filtering if necessary
        if len(control_points) > CONTROL_POINT_FILTER_THRESHOLD:
            if verbose:
                print(
                    f'\t- Control point filtering triggered. (CP count: {len(control_points)})'
                )
            hist = skimage.exposure.histogram(control_point_scores)
            threshold = skimage.filters.threshold_otsu(hist=hist)
            control_points = list(
                filter(lambda cp: cp.score > threshold, control_points)
            )
        cp_count = len(control_points)
        if verbose:
            print(f'\t- Final number of control points: {cp_count}')
            print(f'Done ({(time.time() - t0):.4f} s)')

        # compute lengths and angles between each control points
        if verbose:
            t0 = time.time()
            print(f'\nComputing line lengts and angles... ')
        line_lengths_matrix = np.zeros((cp_count, cp_count))
        line_angles_matrix = np.zeros((cp_count, cp_count))
        for i in range(cp_count):
            for j in range(i, cp_count):
                # this is quite inefficient, but if speed is required,
                # don't use python in the first place
                if i == j:
                    continue
                p1 = control_points[i]
                p2 = control_points[j]
                distance = math.dist([p1.x, p1.y], [p2.x, p2.y])
                line_lengths_matrix[i, j] = distance
                line_lengths_matrix[j, i] = distance
                # the angle isn't in degrees nor radians
                # 0.5 = horizontal, 0.0 and 1.0 = vertical
                angle = math.atan2((p1.y - p2.y), (p1.x - p2.x)) / np.pi
                if angle < 0:
                    angle += 1.0
                line_angles_matrix[i, j] = angle
                line_angles_matrix[j, i] = angle
        angle_avg = np.median(line_angles_matrix)
        length_avg = np.mean(line_lengths_matrix)
        if verbose:
            print(f'\t- average angle: {angle_avg}')
            print(f'\t- average length: {length_avg}')
            print(f'Done ({(time.time() - t0):.4f} s)')

        # compute the brightness cost and angle cost matrices
        if verbose:
            t0 = time.time()
            print(f'\nComputing brigtness and angle cost matrices... ')
        line_brightness_cost_matrix = np.zeros((cp_count, cp_count))
        line_angles_cost_matrix = np.zeros((cp_count, cp_count))
        for i in range(len(control_points)):
            for j in range(i, len(control_points)):
                # this is quite inefficient, but if speed is required,
                # don't use python in the first place
                if i == j:
                    continue
                p1 = control_points[i]
                p2 = control_points[j]
                length = line_lengths_matrix[i, j]
                angle = line_angles_matrix[i, j]
                s1 = abs(angle - angle_avg) * length / 5
                s2 = abs(angle - angle_avg + 0.5)
                s3 = abs(angle - angle_avg - 0.5)
                angle_score = 1 - 2 * min(s1, s2, s3)
                line_angles_cost_matrix[i, j] = angle_score
                line_angles_cost_matrix[j, i] = angle_score
                lmap = np.zeros(blurred.shape)
                rr, cc = skimage.draw.line(p1.x, p1.y, p2.x, p2.y)
                lmap[rr, cc] = 1.0
                lmap_blurred = skimage.filters.gaussian(
                    lmap, LINE_SCORE_KERNEL_BLUR_INTENSITY
                )
                lmap_blurred = np.divide(lmap_blurred, np.max(lmap_blurred))
                score = np.sum(masked_contrast * lmap_blurred) / length
                line_brightness_cost_matrix[i, j] = score
                line_brightness_cost_matrix[j, i] = score
        if verbose:
            print(f'Done ({(time.time() - t0):.4f} s)')

        # combine the brightness and angle scores into a final scalar
        if verbose:
            t0 = time.time()
            print(
                f'\nCombining brightness and angle costs into final score...'
            )
        line_final_score_matrix = np.zeros((cp_count, cp_count))
        for i in range(len(control_points)):
            for j in range(i, len(control_points)):
                # this is quite inefficient, but if speed is required,
                # don't use python in the first place
                if i == j:
                    continue
                brightness_score = line_brightness_cost_matrix[i, j]
                angle_score = line_angles_cost_matrix[i, j]
                length = line_lengths_matrix[i, j]
                length_ratio = length / length_avg
                score = brightness_score * angle_score / (1 + length_ratio)
                line_final_score_matrix[i, j] = score
                line_final_score_matrix[j, i] = score
        if verbose:
            print(f'Done ({(time.time() - t0):.4f} s)')

        # construct the line-net with values based on brightness
        linemap_brightness = np.zeros(blurred.shape)
        if verbose:
            t0 = time.time()
            print(f'\nConstructing brightness line-map...')
        for i in range(len(control_points)):
            for j in range(i, len(control_points)):
                # this is quite inefficient, but if speed is required,
                # don't use python in the first place
                if i == j:
                    continue
                p1 = control_points[i]
                p2 = control_points[j]
                rr, cc = skimage.draw.line(p1.x, p1.y, p2.x, p2.y)
                linemap_brightness[rr, cc] = np.maximum(
                    linemap_brightness[rr, cc],
                    line_brightness_cost_matrix[i, j],
                )
        if verbose:
            print(f'Done ({(time.time() - t0):.4f} s)')

        # construct the line-net with values based on angles
        if verbose:
            t0 = time.time()
            print(f'\nConstructing brightness line-map...')
        linemap_angles = np.zeros(blurred.shape)
        for i in range(len(control_points)):
            for j in range(i, len(control_points)):
                if i == j:
                    continue
                p1 = control_points[i]
                p2 = control_points[j]
                rr, cc = skimage.draw.line(p1.x, p1.y, p2.x, p2.y)
                linemap_angles[rr, cc] = np.maximum(
                    linemap_angles[rr, cc], line_angles_cost_matrix[i, j]
                )
        if verbose:
            print(f'Done ({(time.time() - t0):.4f} s)')

        # construct the line-net with values based on final score
        if verbose:
            t0 = time.time()
            print(f'\nConstructing final-score line-map...')
        linemap_finals = np.zeros(blurred.shape)
        for i in range(len(control_points)):
            for j in range(i, len(control_points)):
                if i == j:
                    continue
                p1 = control_points[i]
                p2 = control_points[j]
                rr, cc = skimage.draw.line(p1.x, p1.y, p2.x, p2.y)
                linemap_finals[rr, cc] = np.maximum(
                    linemap_finals[rr, cc], line_final_score_matrix[i, j]
                )
        if verbose:
            print(f'Done ({(time.time() - t0):.4f} s)')

        # construct the control point map
        if verbose:
            t0 = time.time()
            print(f'\nConstructing control point map...')
        cpmap = np.zeros(blurred.shape)
        for cp in control_points:
            cpmap[cp.x, cp.y] = cp.score
            cpmap[cp.x - 1, cp.y] = cp.score
            cpmap[cp.x + 1, cp.y] = cp.score
            cpmap[cp.x, cp.y - 1] = cp.score
            cpmap[cp.x, cp.y + 1] = cp.score
        if verbose:
            print(f'Done ({(time.time() - t0):.4f} s)')

        # black magic for scar detection
        if verbose:
            t0 = time.time()
            print(f'\nUsing black magic for scar-line detection...')
        best_main_line_points = []
        best_main_line_score = -1
        best_main_line_map = np.zeros(blurred.shape)
        for k in range(min(50, len(control_points))):
            main_line_points = []
            lb = k
            rb = k
            connect_lmap = np.zeros(blurred.shape)
            main_line_points.append(k)
            main_line_total_cost = 0
            while True:
                candidates = np.copy(line_final_score_matrix[lb, :])
                for i in range(len(control_points)):
                    if control_points[lb].y >= control_points[i].y:
                        candidates[i] = 0.0
                for i in range(len(control_points)):
                    if abs(line_angles_matrix[lb, i] - angle_avg) > 0.15:
                        candidates[i] = 0.0
                if np.sum(candidates) <= 0.001:
                    break
                max_idx = np.argmax(candidates)
                max_val = candidates[max_idx]
                p1 = control_points[lb]
                p2 = control_points[max_idx]
                main_line_points.append(max_idx)
                rr, cc = skimage.draw.line(p1.x, p1.y, p2.x, p2.y)
                connect_lmap[rr, cc] = np.maximum(
                    linemap_angles[rr, cc], max_val
                )
                main_line_total_cost += line_brightness_cost_matrix[
                    lb, max_idx
                ]
                lb = max_idx
            while True:
                candidates = np.copy(line_final_score_matrix[rb, :])
                # delete points that are on the right
                for i in range(len(control_points)):
                    if control_points[rb].y <= control_points[i].y:
                        candidates[i] = 0.0
                # delete points that are too high or too low
                for i in range(len(control_points)):
                    if abs(line_angles_matrix[rb, i] - angle_avg) > 0.15:
                        candidates[i] = 0.0
                if np.sum(candidates) <= 0.001:
                    break
                max_idx = np.argmax(candidates)
                max_val = candidates[max_idx]
                p1 = control_points[rb]
                p2 = control_points[max_idx]
                main_line_points.insert(0, max_idx)
                rr, cc = skimage.draw.line(p1.x, p1.y, p2.x, p2.y)
                connect_lmap[rr, cc] = np.maximum(
                    linemap_angles[rr, cc], max_val
                )
                main_line_total_cost += line_brightness_cost_matrix[
                    lb, max_idx
                ]
                rb = max_idx
            if main_line_total_cost > best_main_line_score:
                best_main_line_score = main_line_total_cost
                best_main_line_points = main_line_points
                best_main_line_map = connect_lmap
        if verbose:
            print(f'Done ({(time.time() - t0):.4f} s)')

        # arcane arts for stitch detection
        if verbose:
            t0 = time.time()
            print(f'\nUsing arcane arts for stitch detection...')
        distances_from_main_line = np.zeros(len(control_points))
        for i, cp in enumerate(control_points):
            if i in best_main_line_points:
                continue
            dist_matrix = np.zeros(blurred.shape)
            non_zero = np.nonzero(best_main_line_map)
            best_dist = max(blurred.shape)
            for j in range(len(non_zero[0])):
                px = non_zero[0][j]
                py = non_zero[1][j]
                dist = math.dist([px, py], [cp.x, cp.y])
                dist_matrix[px, py] = dist
                if dist < best_dist:
                    best_dist = dist
            distances_from_main_line[i] = best_dist
        main_line_stitch_score = np.zeros(blurred.shape)
        non_zero = np.nonzero(best_main_line_map)
        scores = []
        for i in range(len(non_zero[0])):
            x = non_zero[0][i]
            y = non_zero[1][i]
            score = np.sum(masked[:, y - 4 : y + 4])
            main_line_stitch_score[x, y] = score
            scores.append(score)
        blurmap = skimage.filters.gaussian(main_line_stitch_score, 3)
        main_line_peaks = skimage.feature.peak_local_max(
            blurmap, min_distance=int(max(blurred.shape) / 25)
        )
        intersections = []
        for i, c in enumerate(main_line_peaks):
            score = float(main_line_stitch_score[c[0], c[1]])
            intersections.append(ControlPoint(c[0], c[1], score))
        intersections.sort(key=lambda x: x.score, reverse=True)
        intersections = intersections[
            0 : min(len(intersections), 10)
        ]   # we want only 5, the extras are for mistake reduction

        angles = np.linspace(0.25, 0.75, STITCH_ANGLE_COUNT)
        intersections_tuples = []
        for i, c in enumerate(intersections):
            map = np.zeros(blurred.shape)
            best_score = 0
            best_angle = 0
            best_T = [0, 0]
            best_R = [0, 0]
            for a in angles:
                p1 = c.x
                p2 = c.y
                t1 = 0
                t2 = min(
                    blurred.shape[1] - 1,
                    int(p2 + abs(-p1 + t1) / math.tan(a * math.pi)),
                )
                r1 = blurred.shape[0] - 1
                r2 = min(
                    blurred.shape[1] - 1,
                    int(p2 - abs(p1 - r1) / math.tan(a * math.pi)),
                )
                rr, cc = skimage.draw.line(r1, r2, t1, t2)
                score = np.sum(masked[rr, cc]) / math.dist([r1, r2], [t1, t2])
                if score > best_score:
                    best_score = score
                    best_angle = a
                    best_T = [t1, t2]
                    best_R = [r1, r2]
            intersections_tuples.append(
                (c, best_T, best_R, best_angle, best_score)
            )
        intersections_tuples.sort(key=lambda x: x[4], reverse=True)
        intersections_tuples = intersections_tuples[
            0 : min(5, len(intersections_tuples))
        ]
        if verbose:
            print(f'Done ({(time.time() - t0):.4f} s)')

        if save_fig:
            plt.imshow(img)
            plt.axis('off')
            if plt.gca().get_legend() is not None:
                plt.gca().get_legend().remove()
            plt.savefig(
                f'{img_ID}_original.pdf',
                format='pdf',
                bbox_inches='tight',
                pad_inches=0,
                dpi=300,
            )
            plt.clf()

            plt.imshow(skimage.color.rgb2hsv(img)[:, :, 0], cmap='inferno')
            plt.axis('off')
            if plt.gca().get_legend() is not None:
                plt.gca().get_legend().remove()
            plt.savefig(
                f'{img_ID}_hue.pdf',
                format='pdf',
                bbox_inches='tight',
                pad_inches=0,
                dpi=300,
            )
            plt.clf()

            plt.imshow(skimage.color.rgb2hsv(img)[:, :, 0], cmap='hsv')
            plt.axis('off')
            if plt.gca().get_legend() is not None:
                plt.gca().get_legend().remove()
            plt.savefig(
                f'{img_ID}_hue_hsv.pdf',
                format='pdf',
                bbox_inches='tight',
                pad_inches=0,
                dpi=300,
            )
            plt.clf()

            plt.imshow(skimage.color.rgb2hsv(img)[:, :, 1])
            plt.axis('off')
            if plt.gca().get_legend() is not None:
                plt.gca().get_legend().remove()
            plt.savefig(
                f'{img_ID}_sat.pdf',
                format='pdf',
                bbox_inches='tight',
                pad_inches=0,
                dpi=300,
            )
            plt.clf()

            plt.imshow(skimage.color.rgb2hsv(img)[:, :, 2])
            plt.axis('off')
            if plt.gca().get_legend() is not None:
                plt.gca().get_legend().remove()
            plt.savefig(
                f'{img_ID}_val.pdf',
                format='pdf',
                bbox_inches='tight',
                pad_inches=0,
                dpi=300,
            )
            plt.clf()

            plt.imshow(masked, cmap='inferno')
            plt.axis('off')
            if plt.gca().get_legend() is not None:
                plt.gca().get_legend().remove()
            plt.savefig(
                f'{img_ID}_masked.pdf',
                format='pdf',
                bbox_inches='tight',
                pad_inches=0,
                dpi=300,
            )
            plt.clf()

            plt.imshow(skimage.color.rgb2gray(img), cmap='gray')
            plt.imshow(cpmap, alpha=1.0 * (cpmap > 0))
            plt.axis('off')
            if plt.gca().get_legend() is not None:
                plt.gca().get_legend().remove()
            plt.savefig(
                f'{img_ID}_cp.pdf',
                format='pdf',
                bbox_inches='tight',
                pad_inches=0,
                dpi=300,
            )
            plt.clf()

            plt.imshow(skimage.color.rgb2gray(img), cmap='gray')
            lmp_norm = linemap_finals / np.max(linemap_finals)
            nz = np.nonzero(lmp_norm)
            lmp_norm_a = np.copy(lmp_norm)
            lmp_norm_a[nz] = lmp_norm[nz] * (2 / 3) + (1 / 3)
            plt.imshow(lmp_norm, alpha=lmp_norm_a, cmap='jet')
            plt.axis('off')
            if plt.gca().get_legend() is not None:
                plt.gca().get_legend().remove()
            plt.savefig(
                f'{img_ID}_lmp.pdf',
                format='pdf',
                bbox_inches='tight',
                pad_inches=0,
                dpi=300,
            )
            plt.clf()

            plt.imshow(skimage.color.rgb2gray(img), cmap='gray')
            plt.imshow(
                1.0 * (best_main_line_map > 0),
                alpha=1.0 * (best_main_line_map > 0),
                cmap='inferno',
            )
            plt.axis('off')
            if plt.gca().get_legend() is not None:
                plt.gca().get_legend().remove()
            plt.savefig(
                f'{img_ID}_ml.pdf',
                format='pdf',
                bbox_inches='tight',
                pad_inches=0,
                dpi=300,
            )
            plt.clf()

            plt.imshow(skimage.color.rgb2gray(img), cmap='gray')
            plt.imshow(blurmap, alpha=blurmap / np.max(blurmap), cmap='winter')
            for c in intersections:
                plt.plot(c.y, c.x, 'r*')
            plt.axis('off')
            if plt.gca().get_legend() is not None:
                plt.gca().get_legend().remove()
            plt.savefig(
                f'{img_ID}_cross.pdf',
                format='pdf',
                bbox_inches='tight',
                pad_inches=0,
                dpi=300,
            )
            plt.clf()

            plt.imshow(skimage.color.rgb2gray(img), cmap='gray')
            plt.imshow(
                1.0 * (best_main_line_map > 0),
                alpha=1.0 * (best_main_line_map > 0),
                cmap='inferno',
            )
            stitch_map = np.zeros(blurred.shape)
            for it in intersections_tuples:
                rr, cc = skimage.draw.line(
                    it[1][0], it[1][1], it[2][0], it[2][1]
                )
                stitch_map[rr, cc] = np.maximum(stitch_map[rr, cc], it[4])
            plt.imshow(
                1.0 * (stitch_map > 0),
                alpha=1.0 * (stitch_map > 0),
                cmap='cool',
            )
            plt.axis('off')
            if plt.gca().get_legend() is not None:
                plt.gca().get_legend().remove()
            plt.savefig(
                f'{img_ID}_final.pdf',
                format='pdf',
                bbox_inches='tight',
                pad_inches=0,
                dpi=300,
            )
            plt.clf()

        if verbose:
            print(f"final scar polyline:")
            for c in best_main_line_points:
                # c je index do control_points
                # control_points je list objektů ControlPoint (z utils)
                print(f"\t[{control_points[c].x}, {control_points[c].y}]")
            print(f"final intersections:")
            for c in intersections_tuples:
                # c je tuple, který obsahuje (v tomhle pořadí):
                # 0 -> ControlPoint, který je samotný průsečík
                # 1 -> bod T, který je jedním z okrajových bodů úsečky, která je steh
                #      většinou je na horním nebo dolním okraji obrázku
                # 2 -> bod R, který je druhým bodem úsečky stehu (spolu s T tvoří úsečku stehy)
                # body R a T jsou dvouprvková pole [x y]
                # 3 -> úhel úsečky, nicméně v ezoterickém formátu, nejsou to ani úhly, ani radiány
                # 4 -> skóre úsečky, to tě asi tady nebude zajímat
                print(f"\t[{c[0].x}, {c[0].y}]")
            print(f"final stitch lines:")
            for c in intersections_tuples:
                print(f"\t[{c[1][0]}, {c[1][1]}] -- [{c[2][0]}, {c[2][1]}]")

            plt.subplot(4, 2, 1)
            plt.imshow(img)
            plt.title('Original image')

            plt.subplot(4, 2, 2)
            plt.imshow(skimage.color.rgb2hsv(img)[:, :, 0], cmap='inferno')
            # plt.imshow(masked, cmap='inferno')
            plt.title('Hue channel')

            plt.subplot(4, 2, 3)
            # plt.imshow(skimage.color.rgb2hsv(img)[:, :, 1])
            plt.imshow(masked, cmap='inferno')
            plt.title('Masked')

            plt.subplot(4, 2, 4)
            plt.imshow(skimage.color.rgb2gray(img), cmap='gray')
            plt.imshow(cpmap, alpha=1.0 * (cpmap > 0))
            plt.title('Detected control points')

            plt.subplot(4, 2, 5)
            plt.imshow(skimage.color.rgb2gray(img), cmap='gray')
            lmp_norm = linemap_finals / np.max(linemap_finals)
            nz = np.nonzero(lmp_norm)
            lmp_norm_a = np.copy(lmp_norm)
            lmp_norm_a[nz] = lmp_norm[nz] * (2 / 3) + (1 / 3)
            plt.imshow(lmp_norm, alpha=lmp_norm_a, cmap='jet')
            plt.title('Constructed line map')

            plt.subplot(4, 2, 6)
            plt.imshow(skimage.color.rgb2gray(img), cmap='gray')
            plt.imshow(
                1.0 * (best_main_line_map > 0),
                alpha=1.0 * (best_main_line_map > 0),
                cmap='inferno',
            )
            plt.title('Detected main line')

            plt.subplot(4, 2, 7)
            plt.imshow(skimage.color.rgb2gray(img), cmap='gray')
            plt.imshow(blurmap, alpha=blurmap / np.max(blurmap), cmap='winter')
            for c in intersections:
                plt.plot(c.y, c.x, 'r*')
            plt.title('Detected crossections')

            plt.subplot(4, 2, 8)
            plt.imshow(skimage.color.rgb2gray(img), cmap='gray')
            plt.imshow(
                1.0 * (best_main_line_map > 0),
                alpha=1.0 * (best_main_line_map > 0),
                cmap='inferno',
            )
            stitch_map = np.zeros(blurred.shape)
            for it in intersections_tuples:
                rr, cc = skimage.draw.line(
                    it[1][0], it[1][1], it[2][0], it[2][1]
                )
                stitch_map[rr, cc] = np.maximum(stitch_map[rr, cc], it[4])
            plt.imshow(stitch_map, alpha=1.0 * (stitch_map > 0), cmap='winter')
            plt.title('Final result')
            plt.suptitle(f'Img : {path}')

            plt.show()


    #----------------------------------------------------------------------------
    incision_polyline = list()
    for c in best_main_line_points:
        incision_polyline.append([int(control_points[c].x), int(control_points[c].y)])

    crossing_angles, crossing_positions = find_angles(best_main_line_points, control_points, intersections_tuples)
    # ----------------------------------------------------------------------------
    return incision_polyline, crossing_positions, crossing_angles

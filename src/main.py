import xmltodict
from matplotlib import pyplot as plt
from utils import *
import skimage
import math
import scipy

ANNOTATION_PATH = './data/annotations.xml'
IMAGES_PATH = './data/images/default/'
GOOD_IMAGES_ID = [14, 17, 18, 20, 24, 28, 33, 38, 44, 45, 52, 56, 57]

with open(ANNOTATION_PATH) as f:
    doc = xmltodict.parse(f.read())

imgs_annotated = doc['annotations']['image']

# explore_rgb_channels(list(range(50)), imgs_annotated, IMAGES_PATH)
# explore_blobs(list(range(50)), imgs_annotated, IMAGES_PATH)
# explore_hsv_channels(list(range(100)), imgs_annotated, IMAGES_PATH)
# explore_thresholding(GOOD_IMAGES_ID, imgs_annotated, IMAGES_PATH, 10)

for img_ID in list(range(100)):
    # img_ID = 22
    print(f'img id: {img_ID}')
    img_fname = imgs_annotated[img_ID]['@name']
    img = skimage.io.imread(IMAGES_PATH + img_fname)

    blur_intensity: int = 2
    point_min_distance: int = 1
    control_points: list[ControlPoint] = []
    lines: list[ControlPointLine] = []
    control_point_score_kernel = get_gauss_kernel(5)

    xy, blurred, masked, masked_contrast = get_control_points(
        img, blur_intensity, point_min_distance
    )

    if len(xy) < 3:
        print('skipped img ID: {img_ID}, not enough control points detected')
        continue

    control_point_scores = np.zeros(len(xy))
    for i, c in enumerate(xy):
        score = compute_score(masked, control_point_score_kernel, c[0], c[1])
        control_point_scores[i] = score
        control_points.append(ControlPoint(c[0], c[1], score))
    control_points.sort(key=lambda x: x.score, reverse=True)
    if len(control_points) > 100:
        print(f'filtering from: {len(control_points)}')
        hist = skimage.exposure.histogram(control_point_scores)
        threshold = skimage.filters.threshold_otsu(hist=hist)
        control_points = list(filter(lambda cp: cp.score > threshold, control_points))

    print(f'number of control points: {len(control_points)}')

    line_lengths_matrix = np.zeros((len(control_points), len(control_points)))
    line_angles_matrix = np.zeros((len(control_points), len(control_points)))
    for i in range(len(control_points)):
        for j in range(i, len(control_points)):
            if i == j:
                continue
            p1 = control_points[i]
            p2 = control_points[j]
            distance = math.dist([p1.x, p1.y], [p2.x, p2.y])
            line_lengths_matrix[i, j] = distance
            line_lengths_matrix[j, i] = distance
            angle = math.atan2((p1.y - p2.y), (p1.x - p2.x)) / np.pi
            if angle < 0:
                angle += 1.0
            line_angles_matrix[i, j] = angle
            line_angles_matrix[j, i] = angle

    angle_avg = np.median(line_angles_matrix)
    # print(f'angle average: {angle_avg}')
    length_avg = np.mean(line_lengths_matrix)
    # print(f'length average: {length_avg}')

    line_brightness_cost_matrix = np.zeros(
        (len(control_points), len(control_points))
    )
    line_angles_cost_matrix = np.zeros(
        (len(control_points), len(control_points))
    )
    for i in range(len(control_points)):
        for j in range(i, len(control_points)):
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
            lmap_blurred = skimage.filters.gaussian(lmap, 1)
            lmap_blurred = np.divide(lmap_blurred, np.max(lmap_blurred))

            score = np.sum(masked_contrast * lmap_blurred) / length
            line_brightness_cost_matrix[i, j] = score
            line_brightness_cost_matrix[j, i] = score

    line_final_score_matrix = np.zeros(
        (len(control_points), len(control_points))
    )
    for i in range(len(control_points)):
        for j in range(i, len(control_points)):
            if i == j:
                continue
            brightness_score = line_brightness_cost_matrix[i, j]
            angle_score = line_angles_cost_matrix[i, j]
            length = line_lengths_matrix[i, j]
            length_ratio = length / length_avg
            score = brightness_score * angle_score / (1 + length_ratio)
            line_final_score_matrix[i, j] = score
            line_final_score_matrix[j, i] = score

    linemap_brightness = np.zeros(blurred.shape)
    for i in range(len(control_points)):
        for j in range(i, len(control_points)):
            if i == j:
                continue
            p1 = control_points[i]
            p2 = control_points[j]
            rr, cc = skimage.draw.line(p1.x, p1.y, p2.x, p2.y)
            linemap_brightness[rr, cc] = np.maximum(
                linemap_brightness[rr, cc], line_brightness_cost_matrix[i, j]
            )

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

    cpmap = np.zeros(blurred.shape)
    for cp in control_points:
        cpmap[cp.x, cp.y] = cp.score
        cpmap[cp.x - 1, cp.y] = cp.score
        cpmap[cp.x + 1, cp.y] = cp.score
        cpmap[cp.x, cp.y - 1] = cp.score
        cpmap[cp.x, cp.y + 1] = cp.score

    for i, cp in enumerate(control_points):
        cp.horizontality = float(1.0 - 2.0 * (np.mean(line_angles_matrix[i, :]) - angle_avg))

    best_main_line_points = []
    best_main_line_score = -1
    best_main_line_map = np.zeros(blurred.shape)
    for k in range(min(50, len(control_points))):
        main_line_points = []
        cp_max = control_points[k]
        # print(f"starting point: {cp_max}")
        lb = k
        rb = k
        connect_lmap = np.zeros(blurred.shape)
        main_line_points.append(cp_max)
        main_line_total_cost = 0
        while True:
            candidates = np.copy(line_final_score_matrix[lb, :])
            # delete points that are on the right
            for i in range(len(control_points)):
                if control_points[lb].y >= control_points[i].y:
                    candidates[i] = 0.0
            # delete points that are too high or too low
            for i in range(len(control_points)):
                if abs(line_angles_matrix[lb, i] - angle_avg) > 0.15:
                    candidates[i] = 0.0
            if np.sum(candidates) <= 0.001:
                break
            max_idx = np.argmax(candidates)
            max_val = candidates[max_idx]
            p1 = control_points[lb]
            p2 = control_points[max_idx]
            main_line_points.append(cp_max)
            rr, cc = skimage.draw.line(p1.x, p1.y, p2.x, p2.y)
            connect_lmap[rr, cc] = np.maximum(linemap_angles[rr, cc], max_val)
            main_line_total_cost += line_brightness_cost_matrix[lb, max_idx]
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
            main_line_points.append(cp_max)
            rr, cc = skimage.draw.line(p1.x, p1.y, p2.x, p2.y)
            connect_lmap[rr, cc] = np.maximum(linemap_angles[rr, cc], max_val)
            main_line_total_cost += line_brightness_cost_matrix[lb, max_idx]
            rb = max_idx
        # print(f"total main line cost: {main_line_total_cost}")
        if main_line_total_cost > best_main_line_score:
            best_main_line_score = main_line_total_cost
            best_main_line_points = main_line_points
            best_main_line_map = connect_lmap

    plt.imshow(img)
    plt.imshow(
        1.0 * (best_main_line_map > 0.0),
        cmap='inferno',
        alpha=1.0 * (best_main_line_map > 0.0),
    )
    plt.imshow(cpmap, cmap='winter', alpha=cpmap / np.max(cpmap))
    plt.show()

    # line_scores: list[float] = []
    # line_angles: list[float] = []
    # line_lengths: list[float] = []
    # sat_channel = skimage.color.rgb2hsv(img)[:, :, 1]   # the sharp one
    # for i in range(len(xy)):
    #     for j in range(i, len(xy)):
    #         if i == j:
    #             continue
    #         p1 = control_points[i]
    #         p2 = control_points[j]
    #         # rr, cc, val = skimage.draw.line_a(p1.x, p1.y, p2.x, p2.y)
    #         rr, cc = skimage.draw.line(p1.x, p1.y, p2.x, p2.y)
    #         keep = 1
    #         map = np.zeros(blurred.shape)
    #         map[rr, cc] = keep
    #         dist = math.dist([p1.x, p1.y], [p2.x, p2.y])
    #         blurred_map = scipy.ndimage.gaussian_filter(map, 1)
    #         blurred_map /= np.max(blurred_map)
    #         score = float(np.sum(masked_contrast * blurred_map) / dist)
    #         line = ControlPointLine(p1, p2, score, map)
    #         lines.append(line)
    #         p1.lines.add(line)
    #         p2.lines.add(line)
    #         line_scores.append(score)
    #         line_angles.append(line.angle)
    #         line_lengths.append(line.length)
    # line_scores /= np.max(line_scores)
    #
    # # keeplines = dict()
    # # for l in lines:
    # #     keeplines[l] = False
    # #
    # # # for each point, keep 4 lines with highest score
    # # for p in control_points:
    # #     sorted_lines = list(p.lines)
    # #     sorted_lines.sort(key=lambda x: x.score, reverse=True)
    # #     for i in range(min(4, len(sorted_lines))):
    # #         keeplines[sorted_lines[i]] = True
    # #
    # # lines = []
    # # for line, keep in keeplines.items():
    # #     if keep:
    # #         lines.append(line)
    #
    # linemap_scores = np.zeros(blurred.shape)
    # linemap_angles = np.zeros(blurred.shape)
    # linemap_final = np.zeros(blurred.shape)
    # # line_score_threshold = np.percentile(line_scores, 50)
    # # print(f'line score threshold: {line_score_threshold}')
    # angle_median = np.median(line_angles)
    # # angle_median = np.average(line_angles, weights=line_scores)
    # line_angle_scores = np.zeros(len(lines))
    # print(f'angle median: {angle_median}')
    # final_line_scores = []
    # avg_length = np.mean(line_lengths)
    # print(f'length mean: {avg_length}')
    # brightness_score_mean_normalized = np.mean(
    #     line_scores / np.max(line_scores)
    # )
    # n = 4
    # cost_matrix = np.zeros((len(control_points), len(control_points)))
    # for i, l in enumerate(lines):
    #     # if abs(l.angle - 0) < 0.1:
    #     # final_score = max(0.5, min(1000.0, l.score / abs(l.angle - angle_median)))
    #     # final_score = l.score
    #     length_ratio = l.length / avg_length
    #     s1 = abs(l.angle - angle_median) * l.length / 5
    #     s2 = abs(l.angle - angle_median + 0.5)   # * math.sqrt(l.length)
    #     s3 = abs(l.angle - angle_median - 0.5)   # * math.sqrt(l.length)
    #     angle_score = 1 - 2 * min(s1, s2, s3)
    #     # angle_score = 1 - 2 * s1
    #     # brightness_score = pow(l.score / np.max(line_scores), n) / pow(
    #     #     brightness_score_mean_normalized, n - 1
    #     # )
    #     brightness_score = l.score / np.max(line_scores)
    #     final_score = angle_score * brightness_score / (length_ratio + 1)
    #     final_line_scores.append(final_score)
    #     linemap_scores = np.maximum(linemap_scores, l.map * l.score)
    #     linemap_angles = np.maximum(linemap_angles, l.map * angle_score)
    #     linemap_final = np.maximum(linemap_final, l.map * final_score)
    #     line_angle_scores[i] = angle_score
    #
    #
    # # plt.subplot(4, 1, 1)
    # # plt.imshow(img)
    # # plt.title('original')

    # plt.subplot(3, 1, 1)
    # plt.imshow(img)
    # plt.imshow(
    #     linemap_brightness,
    #     cmap='inferno',
    #     alpha=1.0 * (linemap_brightness / np.max(linemap_brightness) > 0),
    # )
    # # plt.title('line brightness score')
    # plt.imshow(cpmap, cmap='winter', alpha=cpmap / np.max(cpmap))
    #
    # # plt.subplot(3, 2, 2)
    # # plt.hist(line_scores, bins=30)
    #
    # plt.subplot(3, 1, 2)
    # plt.imshow(img)
    # plt.imshow(
    #     linemap_angles,
    #     cmap='inferno',
    #     alpha=1.0 * (linemap_angles / np.max(linemap_angles) > 0),
    # )
    # # plt.title('line angle score')
    # plt.imshow(cpmap, cmap='winter', alpha=cpmap / np.max(cpmap))
    #
    # # plt.subplot(3, 1, 4)
    # # plt.hist(line_scores, bins=30)
    #
    # plt.subplot(3, 1, 3)
    # plt.imshow(img)
    # plt.imshow(
    #     linemap_finals,
    #     cmap='inferno',
    #     # alpha=1.0 * (linemap_finals / np.max(linemap_finals) > 0),
    #     alpha=linemap_finals / np.max(linemap_finals),
    # )
    # # plt.title('line total score')
    # plt.imshow(cpmap, cmap='winter', alpha=cpmap / np.max(cpmap))
    #
    # # plt.subplot(3, 2, 6)
    # # plt.hist(final_line_scores, bins=30)
    #
    # plt.show()

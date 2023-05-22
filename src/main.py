import xmltodict
from matplotlib import pyplot as plt
from utils import *
import time

ANNOTATION_PATH = './data/annotations.xml'
IMAGES_PATH = './data/images/default/'

with open(ANNOTATION_PATH) as f:
    doc = xmltodict.parse(f.read())

imgs_annotated = doc['annotations']['image']

# explore_rgb_channels(list(range(50)), imgs_annotated, IMAGES_PATH)
# explore_blobs(list(range(50)), imgs_annotated, IMAGES_PATH)
# explore_hsv_channels(list(range(100)), imgs_annotated, IMAGES_PATH)
# explore_thresholding(GOOD_IMAGES_ID, imgs_annotated, IMAGES_PATH, 10)

def run_find_incision(path: str, save_fig: bool, verbose: bool):
    t0 = time.now()
    print(f"Loading image {path}...", end="")
    img = skimage.io.imread(path)
    print(f"Done {time.now() - t0}")

for img_ID in list(range(200)):
    print(f'img id: {img_ID}')
    img_fname = imgs_annotated[img_ID]['@name']
    img = skimage.io.imread(IMAGES_PATH + img_fname)

    # hsv = skimage.color.rgb2hsv(img)
    # plt.imshow(hsv[:, :, 0], cmap='hsv')
    # plt.show()

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
        control_points = list(
            filter(lambda cp: cp.score > threshold, control_points)
        )

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
    # angle_median = np.median(line_angles)

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

    best_main_line_points = []
    best_main_line_score = -1
    best_main_line_map = np.zeros(blurred.shape)
    for k in range(min(50, len(control_points))):
        main_line_points = []
        cp_max = control_points[k]
        lb = k
        rb = k
        connect_lmap = np.zeros(blurred.shape)
        main_line_points.append(k)
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
            main_line_points.append(max_idx)
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
            main_line_points.insert(0, max_idx)
            rr, cc = skimage.draw.line(p1.x, p1.y, p2.x, p2.y)
            connect_lmap[rr, cc] = np.maximum(linemap_angles[rr, cc], max_val)
            main_line_total_cost += line_brightness_cost_matrix[lb, max_idx]
            rb = max_idx
        # print(f"total main line cost: {main_line_total_cost}")
        if main_line_total_cost > best_main_line_score:
            best_main_line_score = main_line_total_cost
            best_main_line_points = main_line_points
            best_main_line_map = connect_lmap

    # plt.imshow(img)
    # plt.imshow(
    #     1.0 * (best_main_line_map > 0.0),
    #     cmap='inferno',
    #     alpha=1.0 * (best_main_line_map > 0.0),
    # )
    # plt.imshow(cpmap, cmap='winter', alpha=cpmap / np.max(cpmap))
    # plt.show()

    distances_from_main_line = np.zeros(len(control_points))
    for i, cp in enumerate(control_points):
        if i in best_main_line_points:
            continue
        dist_matrix = np.zeros(blurred.shape)
        non_zero = np.nonzero(best_main_line_map)
        best_coords = [0, 0]
        best_dist = max(blurred.shape)
        for j in range(len(non_zero[0])):
            px = non_zero[0][j]
            py = non_zero[1][j]
            dist = math.dist([px, py], [cp.x, cp.y])
            dist_matrix[px, py] = dist
            if dist < best_dist:
                best_coords = [px, py]
                best_dist = dist
        distances_from_main_line[i] = best_dist

    main_line_stitch_score = np.zeros(blurred.shape)
    non_zero = np.nonzero(best_main_line_map)
    scores = []
    for i in range(len(non_zero[0])):
        x = non_zero[0][i]
        y = non_zero[1][i]
        # kernel = np.zeros(blurred.shape)
        # kernel[:, y] = 1
        # kernel = skimage.filters.gaussian(kernel, 1)
        # kernel = kernel / np.max(kernel)
        # plt.imshow(img)
        # plt.imshow(kernel, alpha=1.0*(kernel>0))
        # plt.show()
        # score = np.sum(scipy.ndimage.convolve(masked, kernel))
        score = np.sum(masked[:, y - 2 : y + 2])
        main_line_stitch_score[x, y] = score
        scores.append(score)
    blurmap = skimage.filters.gaussian(main_line_stitch_score, 3)
    main_line_peaks = skimage.feature.peak_local_max(
        blurmap, min_distance=int(max(blurred.shape) / 25)
    )
    # print(f'main line peaks: {main_line_peaks}')

    intersections = []
    for i, c in enumerate(main_line_peaks):
        score = float(main_line_stitch_score[c[0], c[1]])
        intersections.append(ControlPoint(c[0], c[1], score))
    intersections.sort(key=lambda x: x.score, reverse=True)
    intersection = intersections[
        0 : min(len(intersections), 10)
    ]   # we want only 5, the extras are for mistake reduction
    print(f'intersections: {intersections}')

    angle_count = 20
    angles = np.linspace(0.25, 0.75, angle_count)
    # best_angles = np.zeros(len(intersections))
    # best_angles_scores = np.zeros(len(intersections))
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
            dx = p1
            # t2 = max(int(p2 + dx/math.tan(a*math.pi)))
            plt.imshow(img)
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
            score = (
                np.sum(masked[rr, cc])
                / math.dist([r1, r2], [t1, t2])
            )
            if score > best_score:
                best_score = score
                best_angle = a
                best_T = [t1, t2]
                best_R = [r1, r2]
            # map[rr, cc] = np.maximum(map[rr, cc], score)
        # intersection_tuples.append(())
        intersections_tuples.append((c, best_T, best_R, best_angle, best_score))
    intersections_tuples.sort(key=lambda x: x[4], reverse=True)
    intersections_tuples = intersections_tuples[0:min(5, len(intersections_tuples))]
        # best_angles[i] = best_angle
        # best_angles_scores[i] = best_score
        # print(f'best_score: {best_score}')
        # plt.imshow(map, alpha=1.0 * (map > 0))
        # plt.show()

    plt.imshow(img)
    plt.imshow(1.0*(best_main_line_map>0), alpha=1.0*(best_main_line_map>0), cmap='inferno')
    stitch_map = np.zeros(blurred.shape)
    for it in intersections_tuples:
        rr, cc = skimage.draw.line(it[1][0], it[1][1], it[2][0], it[2][1])
        stitch_map[rr, cc] = np.maximum(stitch_map[rr, cc], it[4])
    plt.imshow(stitch_map, alpha=1.0*(stitch_map>0), cmap='winter')
    plt.show()

    # peakmap = np.zeros(blurred.shape)
    # for c in intersection:
    #     peakmap[c.x, c.y] = 1#c.score
    #     peakmap[c.x + 1, c.y] = 1#c.score
    #     peakmap[c.x - 1, c.y] = 1#c.score
    #     peakmap[c.x, c.y - 1] = 1#c.score
    #     peakmap[c.x, c.y + 1] = 1#c.score
    # # plt.subplot(2, 1, 1)
    # plt.imshow(img)
    # # plt.subplot(2, 1, 2)
    # plt.imshow(blurmap, alpha = blurmap/np.max(blurmap))  # , alpha=1.0 * (main_line_stitch_score > 0)
    # plt.imshow(peakmap, alpha=1.0 * (peakmap > 0), cmap='cool')
    # plt.show()

    # plt.imshow(img)
    # plt.imshow(-dist_matrix, alpha=1.0*(dist_matrix>0), cmap='inferno')
    # plt.plot(cp.y, cp.x, 'r*')
    # plt.plot(best_coords[1], best_coords[0], 'rs')
    # plt.show()
    # print(distances_from_main_line)

    # plt.subplot(4, 2, 1)
    # plt.imshow(img)
    # plt.imshow(cpmap, cmap='winter', alpha=1.0 * (cpmap > 0))
    # plt.subplot(4, 2, 2)
    # plt.imshow(img)
    # map = np.zeros(blurred.shape)
    # for i, d in enumerate(distances_from_main_line):
    #     map[control_points[i].x, control_points[i].y] = d
    #     map[control_points[i].x + 1, control_points[i].y] = d
    #     map[control_points[i].x - 1, control_points[i].y] = d
    #     map[control_points[i].x, control_points[i].y - 1] = d
    #     map[control_points[i].x, control_points[i].y + 1] = d
    # plt.imshow(
    #     1.0 * (best_main_line_map > 0), alpha=1.0 * (best_main_line_map > 0)
    # )
    # plt.imshow(map, alpha=1.0 * (map > 0), cmap='inferno')
    # plt.show()

    stitch_cost_matrix = np.zeros((len(control_points), len(control_points)))
    stitch_line_map_brightness = np.zeros(blurred.shape)
    stitch_line_map_angle = np.zeros(blurred.shape)
    stitch_line_map_final = np.zeros(blurred.shape)
    stitch_brightness_scores = []
    stitch_angle_scores = []
    stitch_final_scores = []
    for i in range(len(control_points)):
        for j in range(i, len(control_points)):
            if (
                i == j
                or i in best_main_line_points
                or j in best_main_line_points
            ):
                continue
            p1 = control_points[i]
            p2 = control_points[j]
            angle = line_angles_matrix[i, j]
            s2 = abs(angle - angle_avg + 0.5)
            s3 = abs(angle - angle_avg - 0.5)
            angle_score = max(s2, s3)
            brightness_score = line_brightness_cost_matrix[i, j]
            final_score = angle_score * brightness_score
            stitch_angle_scores.append(angle_score)
            stitch_brightness_scores.append(brightness_score)
            stitch_final_scores.append(final_score)
            rr, cc = skimage.draw.line(p1.x, p1.y, p2.x, p2.y)
            stitch_line_map_angle[rr, cc] = np.maximum(
                angle_score, stitch_line_map_angle[rr, cc]
            )
            stitch_line_map_brightness[rr, cc] = np.maximum(
                brightness_score, stitch_line_map_brightness[rr, cc]
            )
            stitch_line_map_final[rr, cc] = np.maximum(
                final_score, stitch_line_map_final[rr, cc]
            )

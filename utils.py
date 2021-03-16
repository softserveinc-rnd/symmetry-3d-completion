import numpy as np
import open3d as o3d
import copy
import time
from sklearn.decomposition import PCA


def get_bounding_box_center(pcd):
    """
    Function calculates the oriented bounding box and center of the given point cloud
    :param pcd: open3D PointCloud object
    :return: tuple - center and bounding box
    """
    obb = pcd.get_oriented_bounding_box()
    return obb.get_center(), obb


def get_convex_hull_lines(pcd):
    """
    Function calculates the convex hull lines of the given point cloud
    :param pcd:  open3D PointCloud object
    :return:  open3D LineSet object
    """
    hull, _ = pcd.compute_convex_hull()
    hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
    return hull_ls


def one_side(p, points, u):
    """

    :param p:
    :param points:
    :param u:
    :return:
    """
    v = p - points
    return np.sign(np.dot(u, v))


def find_reflection(p, points, u):
    """

    :param p:
    :param points:
    :param u:
    :return:
    """
    d = np.dot(points, u)
    return p - 2 * (np.dot(p, u) - d) * u


def get_mirrored_pcd(points, u, coords):
    side = np.array([one_side(p[:3], points, u) < 0 for p in coords])
    coords = np.array(coords)
    result = coords[side]
    right_side = np.array([find_reflection(p, points, u) for p in result])

    another_side = np.array([one_side(p[:3], points, u) > 0 for p in coords])
    another_result = coords[another_side]
    left_side = np.array([find_reflection(p, points, u) for p in another_result])
    repaired = np.concatenate((left_side, right_side), axis=0)
    return repaired

def find_normal(points):
    """

    :param points:
    :return:
    """
    p1, p2, p3 = points[0], points[1], points[2]
    v1 = p1 - p2
    v2 = p2 - p3
    n = np.cross(v1, v2)
    eps = 0.0001
    return n + eps


def normalize(x):
    """
    Function normalizes the list
    :param x: list
    :return: np.array
    """
    x = np.array(x)
    norm = x / np.linalg.norm(x)
    return norm


def centeroidnp(arr):
    """
    Functions calculates the centroid of point cloud points
    :param arr: no.array, point cloud points
    :return: np.array, centroid point
    """
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    sum_z = np.sum(arr[:, 2])
    return np.array([sum_x/length, sum_y/length, sum_z/length])


# def get_normal_candidates(pcd):
#     """
#     Function returns the candidates to be the normal to the symmetry plane
#     :param pcd: open3D PointCloud() object
#     :return: tuple - np.array, np.array, candidates and bounding box centroid
#     """
#     center_bb, bb = get_bounding_box_center(pcd)
#     voxel_size = 0.01
#     pcd_down = pcd.voxel_down_sample(voxel_size)
#     hull_ls = get_convex_hull_lines(pcd_down)
#     vectors = []
#
#     for i in range(len(hull_ls.lines)):
#         line_coords = hull_ls.get_line_coordinate(i)
#         n = normalize(line_coords[1] - line_coords[0])
#         vectors.append(n)
#
#     # center = np.array([0.0, 0.0, 0.0])
#     directions = np.array(vectors)
#     diameter = np.concatenate((directions, -directions), axis=0)
#     dirs = o3d.geometry.PointCloud()
#     dirs.points = o3d.utility.Vector3dVector(diameter)
#
#     with o3d.utility.VerbosityContextManager(
#             o3d.utility.VerbosityLevel.Debug) as cm:
#         labels = np.array(
#             dirs.cluster_dbscan(eps=0.02, min_points=10, print_progress=False))    #  dirs.cluster_dbscan(eps=0.02, min_points=10, print_progress=True))
#
#     max_label = labels.max()
#     clusters_centers = []
#
#     for i in range(max_label + 1):
#         label = labels == i
#         cluster = np.asarray(dirs.points)[label]
#         clusters_centers.append(centeroidnp(cluster))
#
#     corrs = []
#     for i in range(max_label + 1):
#         for j in range(max_label + 1):
#             dist = np.linalg.norm(clusters_centers[i] - clusters_centers[j])
#             if dist > 1.98 and dist < 2.02 and (
#                     np.linalg.norm((clusters_centers[i] + clusters_centers[j]) / 2) < 0.001):
#                 corrs.append((i, j))
#
#     us = []
#     for corr in corrs:
#         u = normalize(clusters_centers[corr[0]] - clusters_centers[corr[1]])
#         us.append(u)
#
#     return us, center_bb


def get_best_normal(candidates, center_bb, pcd):
    """
    Function returns the best candidate for the symmetry plane normal based on the proposed metric
    :param candidates: np.array, candidates
    :param center_bb: np.array, centroid of the bounding box
    :param pcd: open3D PointCloud() object
    :return:  np.array, best candidate
    """
    min_diff = 1000000000000000
    best_u = None

    points_red = np.asarray(pcd.points)
    zeros = np.zeros((points_red.shape[0], 1))
    points_red = np.append(points_red, zeros, axis=1)

    for u in candidates:
        mirrored_points = get_mirrored_pcd(center_bb, u, np.asarray(pcd.points))

        points_blue = mirrored_points
        ones = np.ones((points_blue.shape[0], 1))
        points_blue = np.append(points_blue, ones, axis=1)

        diff = compare_fit(points_red, points_blue)

        if diff < min_diff:
            min_diff = diff
            best_u = u

    return best_u


def find_missing_part(pcd, center, u):
    """
    Function returns the points which complete the damaged region
    :param pcd: open3D PointCloud() object
    :param center: np.array, centroid of the bounding box
    :param u: np.array, best symmetry plane normal candidate
    :return: list, damaged region completed points
    """
    mirrored_points = get_mirrored_pcd(center, u, np.asarray(pcd.points))

    points_red = np.asarray(pcd.points)
    zeros = np.zeros((points_red.shape[0], 1))
    points_red = np.append(points_red, zeros, axis=1)

    points_blue = np.asarray(mirrored_points)
    ones = np.ones((points_blue.shape[0], 1))
    points_blue = np.append(points_blue, ones, axis=1)

    result = np.concatenate((points_red, points_blue), axis=0)
    result_pc = o3d.geometry.PointCloud()
    result_pc.points = o3d.utility.Vector3dVector(result[:, :-1])

    damaged = []
    pcd_tree = o3d.geometry.KDTreeFlann(result_pc)
    for i in range(len(result)):
        [_, idx, _] = pcd_tree.search_knn_vector_3d(result_pc.points[i], 6)
        colors = []
        for id in idx:
            colors.append(int(result[id][-1]))
        if colors.count(1) > 5:
            damaged.append(result[i][:3])

    return damaged


# def get_normal_candidates(pcd):
#     """
#     Function calculates the PCA of the point cloud points
#     :param pcd: open3D PointCloud() object
#     :return: np.array, Principal Components of the point cloud points matix
#     """
#     X = np.asarray(copy.deepcopy(pcd.points))
#     X[:, 0] = X[:, 0] - X[:, 0].mean()
#     X[:, 1] = X[:, 1] - X[:, 1].mean()
#     X[:, 2] = X[:, 2] - X[:, 2].mean()
#     u, s, vh = np.linalg.svd(X, full_matrices=False)
#
#     return vh  # was vh

def get_normal_candidates(pcd):
    pcd.estimate_normals()
    vectors = []

    # for i in range(len(hull_ls.lines)):
    #     line_coords = hull_ls.get_line_coordinate(i)
    #     n = normalize(line_coords[1] - line_coords[0])
    #     vectors.append(n)

    for i in range(len(pcd.normals)):
        vectors.append(np.asarray(pcd.normals[i]))

    center = np.array([0.0, 0.0, 0.0])

    directions = np.array(vectors)

    diameter = np.concatenate((directions, -directions), axis=0)

    pca = PCA(3)
    pca.fit(diameter)

    return pca.components_


# def get_normal_candidates(pcd):
#     """
#     Function calculates the PCA of the point cloud points
#     :param pcd: open3D PointCloud() object
#     :return: np.array, Principal Components of the point cloud points matix
#     """
#     X = np.asarray(copy.deepcopy(pcd.points))
#     pca = PCA(n_components=3)
#     pca.fit(X)
#     return pca.components_.T  # was vh


def get_red_blue_representation(orig_points, mirror_points):
    """
    Function prepares the points to fit the input of metric calcuation
    :param orig_points: ndarray, original point cloud points
    :param mirror_points: ndarray, mirrored point cloud points
    :return: tuple - np.array, np.array, updated (colored) points
    """
    points_red = np.asarray(orig_points)
    zeros = np.zeros((points_red.shape[0], 1))
    points_red = np.append(points_red, zeros, axis=1)

    points_blue = np.asarray(mirror_points)
    ones = np.ones((points_blue.shape[0], 1))
    points_blue = np.append(points_blue, ones, axis=1)

    return points_red, points_blue


def preprocess_point_cloud(pcd, voxel_size):
    """
    Function can downsample point cloud and computes FPFH features
    :param pcd: open3D PointCloud() object
    :param voxel_size: float, voxel size
    :return:  open3D PointCloud() object, np.array
    """
    pcd_down = pcd.voxel_down_sample(voxel_size)
    # pcd_down.estimate_normals()
    # pcd_down = pcd

    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


# def prepare_dataset(voxel_size, filename):
#     """
#     Function prepared original and mirrored point clouds with their FPFH features
#     :param voxel_size: float
#     :param filename: string, point cloud file, .pcd extension
#     :return:
#     """
#     print(":: Load two point clouds and disturb initial pose.")
#     # source = o3d.io.read_point_cloud("../generate_dataset/small_damages/plane_many_damages.pcd")
#     # target = o3d.io.read_point_cloud("../generate_dataset/small_damages/plane_many_damages.pcd")
#
#     # gt = o3d.io.read_point_cloud(filename, format='pcd')
#     # source = generate_damages_point_cloud(gt, 0.3, 10)
#
#     source = o3d.io.read_point_cloud(filename, format='pcd')
#
#     center, bb = get_bounding_box_center(source)
#     symmetry_normal_svd = get_best_candidate(source)
#     # symmetry_normal_svd = normalize(np.random.random(3))
#     mirrored_points = get_mirrored_pcd(center, symmetry_normal_svd, np.asarray(source.points))
#     target = o3d.geometry.PointCloud()
#     target.points = o3d.utility.Vector3dVector(np.array(mirrored_points))
#
#     trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
#                              [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
#     # source.transform(trans_init)
#     target.transform(trans_init)
#     # draw_registration_result(source, target, np.identity(4))
#
#     source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
#     target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
#     return source, target, source_down, target_down, source_fpfh, target_fpfh


def prepare_dataset(voxel_size, filename):
    """
    Function prepared original and mirrored point clouds with their FPFH features
    :param voxel_size: float
    :param filename: string, point cloud file, .pcd extension
    :return:
    """
    # source = o3d.io.read_point_cloud("../generate_dataset/small_damages/plane_many_damages.pcd")
    # target = o3d.io.read_point_cloud("../generate_dataset/small_damages/plane_many_damages.pcd")

    # gt = o3d.io.read_point_cloud(filename, format='pcd')
    # source = generate_damages_point_cloud(gt, 0.3, 10)

    source = o3d.io.read_point_cloud(filename, format='pcd')

    center, bb = get_bounding_box_center(source)
    # start = time.time()
    symmetry_normal_svd = get_best_candidate(source)
    # end = time.time()
    # print("candidate time: ", end - start)
    # symmetry_normal_svd = normalize(np.random.random(3))
    mirrored_points = get_mirrored_pcd(center, symmetry_normal_svd, np.asarray(source.points))
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(np.array(mirrored_points))

    # trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
    #                          [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    # source.transform(trans_init)
    # target.transform(trans_init)
    # draw_registration_result(source, target, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh, symmetry_normal_svd, center


def prepare_dataset2(voxel_size, filename, symmetry_u):
    """
    Function prepared original and mirrored point clouds with their FPFH features
    :param voxel_size: float
    :param filename: string, point cloud file, .pcd extension
    :return:
    """
    # source = o3d.io.read_point_cloud("../generate_dataset/small_damages/plane_many_damages.pcd")
    # target = o3d.io.read_point_cloud("../generate_dataset/small_damages/plane_many_damages.pcd")

    # gt = o3d.io.read_point_cloud(filename, format='pcd')
    # source = generate_damages_point_cloud(gt, 0.3, 10)

    source = o3d.io.read_point_cloud(filename, format='pcd')

    center, bb = get_bounding_box_center(source)
    start = time.time()
    # symmetry_normal_svd = get_best_candidate(source)
    end = time.time()
    print("candidate time: ", end - start)
    # symmetry_normal_svd = normalize(np.random.random(3))
    mirrored_points = get_mirrored_pcd(center, symmetry_u, np.asarray(source.points))
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(np.array(mirrored_points))

    # trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
    #                          [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    # source.transform(trans_init)
    # target.transform(trans_init)
    # draw_registration_result(source, target, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh, symmetry_u, center


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    """
    Function calculates the transformation to register two point clouds
    :param source_down: downsampled original point cloud
    :param target_down: downsampled mirrored point cloud
    :param source_fpfh: original point cloud FPFH features
    :param target_fpfh: mirrored point cloud FPFH features
    :param voxel_size: float
    :return:
    """
    distance_threshold = voxel_size * 1.5
    result = o3d.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, distance_threshold,
        o3d.registration.TransformationEstimationPointToPoint(False),
        4, [
            o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.registration.RANSACConvergenceCriteria(4000000, 500))
    return result


#===========================!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
# result_ransac !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1

def refine_registration(source, target, result_ransac, voxel_size):
    distance_threshold = voxel_size * 0.4
    result = o3d.registration.registration_icp(
    source, target, distance_threshold, np.eye(4,4),   # result_ransac.transformation
    o3d.registration.TransformationEstimationPointToPlane())
    return result


def get_best_candidate(pcd):
    vh = get_normal_candidates(pcd)
    center, bb = get_bounding_box_center(pcd)
    results = {}

    # print("total = ", len(vh))

    for i in range(len(vh)):
        start = time.time()
        new_u = vh[i]
        mirrored_points = get_mirrored_pcd(center, normalize(new_u), np.asarray(pcd.points))
        points_red, points_blue = get_red_blue_representation(pcd.points, mirrored_points)
        dist = compare_fit(points_red, points_blue)
        end = time.time()
        # print("one candidate candidate took ", end - start)
        results[i] = dist

    return vh[sorted(results.items(), key=lambda item: item[1])[-1][0]]


def metric(pcd1, pcd2):
    points_red, points_blue = get_red_blue_representation(pcd1.points, pcd2.points)
    start = time.time()
    diff = compare_fit(points_red, points_blue) # compare_fit without 3!!!
    print("Calculated end: ", time.time() - start)
    return diff


def compare_fit(points_red, points_blue):
    result = np.concatenate((points_red, points_blue), axis=0)
    result_pc = o3d.geometry.PointCloud()
    result_pc.points = o3d.utility.Vector3dVector(result[:, :-1])

    start = time.time()
    pcd_tree = o3d.geometry.KDTreeFlann(result_pc)
    # proportions = []
    good_points = 0
    for i in range(0, len(result), 10):
        [_, idx, _] = pcd_tree.search_knn_vector_3d(result_pc.points[i], 6)
        colors = []
        for id in idx:
            colors.append(int(result[id][-1]))
        if abs((colors.count(0) / len(colors)) - 0.5) == 0:
            good_points += 1
    end = time.time()
    # print("metric calculation took ", end - start)
    return good_points / points_red.shape[0]


def compare_fit5(points_red, points_blue):
    result = np.concatenate((points_red, points_blue), axis=0)
    result_pc = o3d.geometry.PointCloud()
    result_pc.points = o3d.utility.Vector3dVector(result[:, :-1])

    pcd_tree = o3d.geometry.KDTreeFlann(result_pc)
    # proportions = []
    good_points = 0
    for i in range(len(result)):
        [_, idx, _] = pcd_tree.search_knn_vector_3d(result_pc.points[i], 30)

        # if i == 1:
        #     print("here: ")
        #     print(result_pc.points[i])
        #     print(idx)
        #     for id in idx:
        #         print(result[id][-1], ", dist = ", result_pc.points[i] - result[id][:-1])

        colors = []
        for id in idx:
            if np.linalg.norm(result_pc.points[i] - result[id][:-1]) < 0.01:
                colors.append(int(result[id][-1]))
        # print(len(colors))
        if abs((colors.count(0) / len(colors)) - 0.5) <= 0.0001:
            good_points += 1
    return good_points / points_red.shape[0]


def compare_fit3(points_red, points_blue):
    result = np.concatenate((points_red, points_blue), axis=0)
    result_pc = o3d.geometry.PointCloud()
    result_pc.points = o3d.utility.Vector3dVector(result[:, :-1])
    good_points = 0

    # pcd_tree = o3d.geometry.KDTreeFlann(result_pc)
    # proportions = []

    eps = 0.01
    points = np.asarray(result_pc.points)

    for i in range(len(result)):
        # print(i)
        # [, idx, ] = pcd_tree.search_knn_vector_3d(result_pc.points[i], 6)
        point_current = points[i]
        # indexes = (points - point_current) < np.array(point_current + eps)
        indexes = abs(points - point_current) < eps
        # points_near = points[points < np.array(point_current + eps)]
        # print(indexes[0])
        bools = [i for i in range(len(indexes)) if indexes[i][0] == True and indexes[i][1] == True and indexes[i][2] == True]
        points_near = result[bools]

        # print(len(points_near))
        colors = []
        # np.reshape(points_near, (-1, 3))
        for p in points_near:
            colors.append(int(p[-1]))
        if abs((colors.count(0) / len(colors)) - 0.5) == 0:
            good_points += 1

    return good_points / points_red.shape[0]


def compare_fit4(points_red, points_blue):
    result = np.concatenate((points_red, points_blue), axis=0)
    result_pc = o3d.geometry.PointCloud()
    result_pc.points = o3d.utility.Vector3dVector(result[:, :-1])

    pcd_tree = o3d.geometry.KDTreeFlann(result_pc)
    # proportions = []
    good_points = 0
    for i in range(len(result)):
        [_, idx, _] = pcd_tree.search_knn_vector_3d(result_pc.points[i], 10)
        colors = []
        for id in idx:
            colors.append(int(result[id][-1]))
        if abs((colors.count(0) / len(colors)) - 0.5) == 0:
            good_points += 1
    return good_points / points_red.shape[0]
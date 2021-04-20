import numpy as np
import open3d as o3d
import copy
import random
from utils import *


def generate_damages_point_cloud(pcd, damage_rate, n_regions):
    """
    Function generates the damaged point cloud  with n_regions number of damages based on the given point cloud
    :param pcd: open3D PointCloud() object
    :param damage_rate: float (0 < damage_rate < 1)
    :param n_regions: integer, number of damages regions
    :return: open3D PointCloud() object with damages
    """
    pc = copy.deepcopy(pcd)
    pcd_tree = o3d.geometry.KDTreeFlann(pc)
    random_point_ids = random.sample(range(len(pc.points)), n_regions)
    damaged_points = np.asarray(pc.points)
    points_to_delete = []
    for id in random_point_ids:
        [_, idx, _] = pcd_tree.search_knn_vector_3d(pc.points[id], int(len(pc.points) * damage_rate / n_regions))
        points_to_delete.append(np.asarray(idx))
    pc.points = o3d.utility.Vector3dVector(np.delete(damaged_points, np.unique(np.array(points_to_delete)), 0))
    return pc


# def generate_damaged_point_cloud(pc, sample_points, damage_rate):
#     """
#
#     :param pc:
#     :param sample_points:
#     :param damage_rate:
#     :return:
#     """
#     pcd_tree = o3d.geometry.KDTreeFlann(pc)
#     random_point_idx = random.randrange(len(pc.points))
#     [_, idx, _] = pcd_tree.search_knn_vector_3d(pc.points[random_point_idx], int(sample_points * damage_rate))
#     pc.points = o3d.utility.Vector3dVector(np.delete(np.asarray(pc.points), idx, 0))
#     return pc


def generate_damages_point_cloud_ideal(pcd, damage_rate, n_regions):
    """
    Function generates the ideal damaged point cloud with no damages intersections
   :param pcd: open3D PointCloud() object
    :param damage_rate: float (0 < damage_rate < 1)
    :param n_regions: integer, number of damages regions
    :return: open3D PointCloud() object with ideal damages
    """
    center, bb = get_bounding_box_center(pcd)
    symmetry_normal_svd = get_best_candidate(pcd)

    side = np.array([one_side(p[:3], center, symmetry_normal_svd) < 0 for p in pcd.points])
    coords = np.array(pcd.points)
    result = coords[side]

    another_side = np.array([one_side(p[:3], center, symmetry_normal_svd) > 0 for p in coords])
    another_result = coords[another_side]

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(np.array(result))

    another_pc = o3d.geometry.PointCloud()
    another_pc.points = o3d.utility.Vector3dVector(np.array(another_result))

    pcd_tree = o3d.geometry.KDTreeFlann(pc)
    random_point_ids = random.sample(range(len(pc.points)), n_regions)
    damaged_points = np.asarray(pc.points)
    points_to_delete = []
    anchor_points_to_delete = []
    for id in random_point_ids:
        [_, idx, _] = pcd_tree.search_knn_vector_3d(pc.points[id], int(len(pc.points) * damage_rate / n_regions))
        anchor_points_to_delete.append(np.sum(np.asarray(pc.points)[np.asarray(idx)], axis=0) / len(np.asarray(idx)))
        points_to_delete.append(np.asarray(idx))

    right_side = np.array([find_reflection(p, center, symmetry_normal_svd) for p in anchor_points_to_delete])

    pcd_tree2 = o3d.geometry.KDTreeFlann(another_pc)
    points_to_delete2 = []
    points_to_delete_coords2 = np.array([[0, 0, 0]])
    for anchor_point in right_side:
        [_, idx2, _] = pcd_tree2.search_knn_vector_3d(anchor_point, int(len(pc.points) * damage_rate / n_regions))
        points_to_delete2.append(np.asarray(idx2))
        points_to_delete_coords2 = np.concatenate((points_to_delete_coords2, np.asarray(another_pc.points)[np.asarray(idx2)]), 0)

    pc.points = o3d.utility.Vector3dVector(np.delete(damaged_points, np.unique(np.array(points_to_delete)), 0))

    another_pc_temp = o3d.geometry.PointCloud()
    another_pc_temp.points = o3d.utility.Vector3dVector(np.delete(np.asarray(another_pc.points), np.unique(np.array(points_to_delete2)), 0))

    pcd_tree3 = o3d.geometry.KDTreeFlann(another_pc_temp)
    random_point_ids3 = random.sample(range(len(another_pc_temp.points)), n_regions)
    damaged_points3 = np.asarray(another_pc_temp.points)
    points_to_delete3 = []
    for id in random_point_ids3:
        [_, idx, _] = pcd_tree3.search_knn_vector_3d(another_pc_temp.points[id], int(len(another_pc_temp.points) * damage_rate / n_regions))
        points_to_delete3.append(np.asarray(idx))

    another_pc.points = o3d.utility.Vector3dVector(np.concatenate((points_to_delete_coords2, np.delete(damaged_points3, np.unique(np.array(points_to_delete3)), 0)), 0))

    # o3d.visualization.draw_geometries(
    #     [pc.paint_uniform_color([0.5, 0.5, 0.5]), another_pc.paint_uniform_color([1, 0, 0])])

    result_damaged_pc = o3d.geometry.PointCloud()
    result_damaged_pc.points = o3d.utility.Vector3dVector(np.concatenate((np.asarray(another_pc.points), np.asarray(pc.points)), 0))
    return result_damaged_pc
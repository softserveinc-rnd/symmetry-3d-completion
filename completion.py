import getopt
import sys
from utils import *


def visualize_sphere(pcd):
    center_bb, bb = get_bounding_box_center(pcd)

    pcd.points = o3d.utility.Vector3dVector(np.array(pcd.points) - center_bb)

    hull_ls = get_convex_hull_lines(pcd)

    o3d.visualization.draw_geometries(
        [pcd.paint_uniform_color([0.5, 0.5, 0.5]), hull_ls.paint_uniform_color([1, 0, 0])])

    normals_pc = o3d.geometry.PointCloud()
    normals_pc.points = o3d.utility.Vector3dVector(np.asarray(pcd.normals))

    lineset = o3d.geometry.LineSet()
    lineset_normals = lineset.create_from_point_cloud_correspondences(pcd, normals_pc, [(i,i) for i in range(0,len(pcd.points), 50)])

    o3d.visualization.draw_geometries(
        [pcd.paint_uniform_color([0.5, 0.5, 0.5]), lineset_normals.paint_uniform_color([0, 1, 0])])

    vectors = []

    for i in range(len(pcd.normals)):
        vectors.append(np.asarray(pcd.normals[i]))

    center = np.array([0.0, 0.0, 0.0])

    directions = np.array(vectors)

    diameter = np.concatenate((directions, -directions), axis=0)

    dirs = o3d.geometry.PointCloud()
    dirs.points = o3d.utility.Vector3dVector(diameter)

    vectors2 = []
    for i in range(len(hull_ls.lines)):
        line_coords = hull_ls.get_line_coordinate(i)
        n = normalize(line_coords[1] - line_coords[0])
        vectors2.append(n)

    directions2 = np.array(vectors2)

    diameter2 = np.concatenate((directions2, -directions2), axis=0)

    dirs2 = o3d.geometry.PointCloud()
    dirs2.points = o3d.utility.Vector3dVector(diameter2)

    pca = PCA(3)
    pca.fit(diameter)

    print(pca.components_)
    print(pca.explained_variance_)

    components = o3d.geometry.PointCloud()
    components.points = o3d.utility.Vector3dVector(pca.components_)

    center_pc = o3d.geometry.PointCloud()
    center_pc.points = o3d.utility.Vector3dVector(np.array([center]))

    lineset_one = lineset.create_from_point_cloud_correspondences(center_pc, components, [(0, 0), (0, 1), (0, 2)])

    pca.fit(diameter2)

    components3 = o3d.geometry.PointCloud()
    components3.points = o3d.utility.Vector3dVector(pca.components_)

    lineset_three = lineset.create_from_point_cloud_correspondences(center_pc, components3, [(0, 0), (0, 1), (0, 2)])

    o3d.visualization.draw_geometries([pcd, dirs2.paint_uniform_color([0.5, 0.5, 0.5]),
                                       components3.paint_uniform_color([1, 0, 0]),
                                       lineset_three.paint_uniform_color([1, 0, 0]),
                                       hull_ls.paint_uniform_color([1, 0, 0])])

    o3d.visualization.draw_geometries([pcd, dirs.paint_uniform_color([0.5, 0.5, 0.5]),
                                       components.paint_uniform_color([0, 1, 0]),
                                       lineset_one.paint_uniform_color([0, 1, 0]), lineset_normals.paint_uniform_color([0,1,0])])

    o3d.visualization.draw_geometries([pcd, components.paint_uniform_color([0, 1, 0]),
                                       lineset_one.paint_uniform_color([0, 1, 0]),
                                       components3.paint_uniform_color([1, 0, 0]),
                                       lineset_three.paint_uniform_color([1, 0, 0])])


def complete(filename, visualizations=False, time_report=False):
    voxel_size = 0.05

    pcd = o3d.io.read_point_cloud(filename, format='pcd')
    start = time.time()

    pcd.estimate_normals()

    start2 = time.time()
    if visualizations:
        o3d.visualization.draw_geometries([pcd.paint_uniform_color([0.5, 0.5, 0.5])])
        visualize_sphere(pcd)
    end2 = time.time() - start2

    source, target, source_down, target_down, source_fpfh, target_fpfh, symmetry_normal_svd, center = prepare_dataset(
        voxel_size, filename)


    result_ransac = execute_global_registration(source_down, target_down,
                                                source_fpfh, target_fpfh,
                                                voxel_size)

    mirrored_pc_bad = copy.deepcopy(target)
    mirrored_pc_bad = mirrored_pc_bad.transform(np.linalg.inv(result_ransac.transformation))

    source.estimate_normals()
    target.estimate_normals()

    result_icp = refine_registration(source, target, result_ransac, voxel_size)

    mirrored_pc = copy.deepcopy(target)
    mirrored_pc = mirrored_pc.transform(np.linalg.inv(result_icp.transformation))

    symmetry_normal_svd_pc = o3d.geometry.PointCloud()
    symmetry_normal_svd_pc.points = o3d.utility.Vector3dVector([symmetry_normal_svd])

    symmetry_normal_svd_pc.transform(np.linalg.inv(result_icp.transformation))

    start3 = time.time()
    if visualizations:
        o3d.visualization.draw_geometries([mirrored_pc.paint_uniform_color([0, 0, 1])])
        o3d.visualization.draw_geometries(
            [source.paint_uniform_color([1, 0, 0]), mirrored_pc_bad.paint_uniform_color([0, 0, 1])])
        o3d.visualization.draw_geometries(
            [source.paint_uniform_color([1, 0, 0]), mirrored_pc.paint_uniform_color([0, 0, 1])])
    end3 = time.time() - start3

    points_red, points_blue = get_red_blue_representation(source.points, mirrored_pc.points)

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

    damaged_pc = o3d.geometry.PointCloud()
    damaged_pc.points = o3d.utility.Vector3dVector(np.array(damaged))

    final_pc = o3d.geometry.PointCloud()
    final_pc.points = o3d.utility.Vector3dVector(
        np.concatenate((np.array(source.points), np.array(damaged_pc.points)), axis=0))

    end = time.time()
    if time_report:
        print("time: ", end - start - end2 - end3)

    o3d.visualization.draw_geometries(
        [source.paint_uniform_color([0.5, 0.5, 0.5]), damaged_pc.paint_uniform_color([0, 1, 0])])

    final_pc.estimate_normals()

    o3d.visualization.draw_geometries([final_pc.paint_uniform_color([0.5, 0.5, 0.5])])

if __name__ == '__main__':
    help_string = """
        -h, --help           print help message
            --filepath       filepath (pcd to complete)
            --visualization  show middle steps
            --time           print time report
        """
    arguments = sys.argv[1:]
    try:
        opts, args = getopt.getopt(arguments, "h", ["help", "filename=", "time=", "visualization="])
    except getopt.GetoptError:
        print("Help")
        sys.exit(2)

    filename = None
    time_report = None
    visualization = None

    for opt, arg in opts:
        if opt in ('-h', "--help"):
            print(help_string)
            sys.exit()
        elif opt in ("--filename"):
            filename = arg
        elif opt in ("--time"):
            time_report = arg
        elif opt in ("--visualization"):
            visualization = arg

    if filename == None:
        print(help_string)
        sys.exit()

    if time_report == "y":
        time_report = True
    else:
        time_report = False

    if visualization == "y":
        visualization = True
    else:
        visualization = False

    complete(filename,visualization, time_report)

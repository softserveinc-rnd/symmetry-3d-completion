import numpy as np
import torch


def distChamfer_cpu(x, y):
    """
    Function calculates the Chamfer distance between two point clouds (using CPU)
    :param x: PointCloud1 points
    :param y: PointCloud2 points
    :return: Chamfer Distance metric

    Usage:
    Considering pc1 and pc2 are open3D PointCloud() objects:
    x = np.asarray(pc1.points)
    y = np.asarray(pc2.points)
    dist = distChamfer_cpu(x, y)

    For the simplicity of the evaluation we compare the distances multiplied by 10^4
    """
    num_points_x, points_dim_x = x.shape
    num_points_y, points_dim_y = y.shape
    xx = np.dot(x, x.T)
    yy = np.dot(y, y.T)
    zz = np.dot(x, y.T)
    diag_ind_x = np.arange(0, num_points_x)
    diag_ind_y = np.arange(0, num_points_y)
    rx = np.tile(np.expand_dims(xx[diag_ind_x, diag_ind_x], axis=1), (1, zz.shape[1]))
    ry = np.tile(np.expand_dims(yy[diag_ind_y, diag_ind_y], axis=0), (zz.shape[0], 1))

    #==========================
    # print("hello")
    # print(rx + ry)
    # sum = rx + ry
    # print(sum)
    # prod = 2 * zz
    # print(prod)
    # P = sum
    #======================
    P = rx + ry - 2 * zz
    # print(P)
    avg_dist_x2y = np.mean(np.min(P, 1))
    avg_dist_y2x = np.mean(np.min(P, 0))

    return avg_dist_x2y + avg_dist_y2x


def distChamfer(x, y):
    """
    Function calculates the Chamfer distance between two point clouds (using GPU)
    :param x: PointCloud1 points
    :param y: PointCloud2 points
    :return: Chamfer Distance metric

    Usage:
    Considering pc1 and pc2 are open3D PointCloud() objects:
    x = np.asarray(pc1.points)
    y = np.asarray(pc2.points)
    dist = distChamfer(torch.from_numpy(np.array([x])).cuda().contiguous(), torch.from_numpy(np.array([y]))
            .cuda().contiguous())
    dist = dist.cpu().numpy()

    For the simplicity of the evaluation we compare the distances multiplied by 10^4
    """
    bs_x, num_points_x, points_dim_x = x.size()
    bs_y, num_points_y, points_dim_y = y.size()
    xx = torch.bmm(x, x.transpose(2, 1))
    yy = torch.bmm(y, y.transpose(2, 1))
    zz = torch.bmm(x, y.transpose(2, 1))
    diag_ind_x = torch.arange(0, num_points_x).type(torch.cuda.LongTensor)
    diag_ind_y = torch.arange(0, num_points_y).type(torch.cuda.LongTensor)
    rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(2).expand_as(zz)
    ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
    P = rx + ry - 2 * zz
    avg_dist_x2y = torch.mean(torch.min(P, 2)[0])
    avg_dist_y2x = torch.mean(torch.min(P, 1)[0])

    return avg_dist_x2y + avg_dist_y2x
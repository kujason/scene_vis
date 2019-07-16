from collections import Counter
import os

import numpy as np
from PIL import Image
import pykitti


def get_raw_data(drive_id, raw_dir):

    # Load raw data
    drive_date = drive_id[0:10]
    drive_num = drive_id[17:21]
    raw_data = pykitti.raw(raw_dir, drive_date, drive_num)

    return raw_data


def get_calib_dir(raw_handler, drive_date):

    kitti_raw_dir = raw_handler.dataset_dir

    return os.path.join(kitti_raw_dir, drive_date)


def get_image_size(sample_obj):
    rgb_path = sample_obj.rgb_path
    image_size = Image.open(rgb_path).size
    return image_size


def read_calib_file(path):
    """Reads calibration file.
    Taken from https://github.com/hunse/kitti

    Args:
        path: path of calibration file

    Returns:
        data: numpy array of calibration information

    """
    float_chars = set("0123456789.e+- ")
    data = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                # try to cast to float array
                try:
                    data[key] = np.array([np.float32(i) for i in value.split(' ')])
                except ValueError:
                    # casting error: data[key] already eq. value, so pass
                    pass

    return data


def read_velodyne_points(velo_path):
    """Reads velodyne points and intensities from file

    Args:
        velo_path: path to velodyne file

    Returns:
        xyzi: (N, 4) xyz points and intensities
    """
    xyzi = np.fromfile(velo_path, dtype=np.float32).reshape(-1, 4)
    return xyzi


def get_velo_points(velo_path, keep_front=False):
    """Gets the velodyne points in the velodyne frame. Intensities are discarded

    Args:
        velo_path: path to velodyne file
        keep_front: optional, whether to only keep front points

    Returns:
        velo_points: (N, 3) points in velodyne frame
    """
    # Get velodyne points
    velo_points = read_velodyne_points(velo_path)[:, 0:3]

    if keep_front:
        # Only keep points with positive x value
        velo_points = velo_points[velo_points[:, 0] > 0]

    return velo_points


def get_cam2cam_calib(calib_dir):
    # Get cam to cam calibration
    cam2cam_calib = read_calib_file(calib_dir + '/calib_cam_to_cam.txt')
    return cam2cam_calib


def get_imu2velo_calib(drive_date_dir):
    imu2velo_calib = read_calib_file(drive_date_dir + '/calib_imu_to_velo.txt')
    return imu2velo_calib


def get_velo2cam_calib(calib_dir):
    # Read velo to cam calibration
    velo2cam_calib = read_calib_file(calib_dir + '/calib_velo_to_cam.txt')
    return velo2cam_calib


def sub2ind(matrix_size, row_sub, col_sub):
    m, n = matrix_size
    return row_sub * (n - 1) + col_sub - 1


def project_velo_to_depth_map(calib_dir, velo_points, im_shape, cam_idx):
    """Projects velodyne points to depth map.
    Adapted from https://github.com/mrharicot/monodepth

    Args:
        calib_dir: calibration file directory
        velo_points: (N, 3) point cloud in velodyne frame
        im_shape: rgb image shape
        cam_idx: camera index (2 or 3)

    Returns:
        depth_map: depth map of the projected points
    """

    # Load calibration files
    cam2cam = get_cam2cam_calib(calib_dir)
    velo2cam = get_velo2cam_calib(calib_dir)
    velo2cam = np.hstack(
        (velo2cam['R'].reshape(3, 3), velo2cam['T'][..., np.newaxis]))
    velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))

    # Get padded r_rect matrix
    r_cam2rect = np.eye(4)
    r_cam2rect[:3, :3] = cam2cam['R_rect_00'].reshape(3, 3)
    r_cam2rect[3, 3] = 1.0

    # Compute rectified projection matrix velodyne->image plane
    p_rect = cam2cam['P_rect_{:02d}'.format(cam_idx)].reshape(3, 4)
    p_velo2im = np.dot(np.dot(p_rect, r_cam2rect), velo2cam)

    # Project the points to the camera
    velo_points_padded = np.pad(velo_points, [0, 1], mode='constant', constant_values=1.0)
    velo_pts_im = np.dot(p_velo2im, velo_points_padded.T).T
    velo_pts_im[:, :2] = velo_pts_im[:, :2] / velo_pts_im[:, 2][..., np.newaxis]

    # Check if in bounds
    velo_pts_im[:, 0] = np.round(velo_pts_im[:, 0])
    velo_pts_im[:, 1] = np.round(velo_pts_im[:, 1])
    val_inds = (velo_pts_im[:, 0] >= 0) & (velo_pts_im[:, 1] >= 0)
    val_inds = val_inds & \
        (velo_pts_im[:, 0] < im_shape[1]) & \
        (velo_pts_im[:, 1] < im_shape[0])
    velo_pts_im = velo_pts_im[val_inds, :]

    # Project to image
    depth = np.zeros(im_shape)
    depth[velo_pts_im[:, 1].astype(np.int32),
          velo_pts_im[:, 0].astype(np.int32)] = velo_pts_im[:, 2]

    # Find the duplicate points and choose the closest depth
    inds = sub2ind(depth.shape, velo_pts_im[:, 1], velo_pts_im[:, 0])
    dupe_inds = [item for item, count in Counter(inds).items() if count > 1]
    for dd in dupe_inds:
        pts = np.where(inds == dd)[0]
        x_loc = int(velo_pts_im[pts[0], 0])
        y_loc = int(velo_pts_im[pts[0], 1])
        depth[y_loc, x_loc] = velo_pts_im[pts, 2].min()
    depth[depth < 0] = 0

    return depth

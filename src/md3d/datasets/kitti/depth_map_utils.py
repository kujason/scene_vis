import cv2
import numpy as np
import png

from md3d.datasets.kitti import calib_utils


def read_depth_map(depth_map_path):

    depth_image = cv2.imread(depth_map_path, cv2.IMREAD_ANYDEPTH)
    depth_map = depth_image / 256.0

    # Discard depths less than 10cm from the camera
    depth_map[depth_map < 0.1] = 0.0

    return depth_map.astype(np.float32)


def save_depth_map(save_path, depth_map,
                   version='cv2', png_compression=3):
    """Saves depth map to disk as uint16 png

    Args:
        save_path: path to save depth map
        depth_map: depth map numpy array [h w]
        version: 'cv2' or 'pypng'
        png_compression: Only when version is 'cv2', sets png compression level.
            A lower value is faster with larger output,
            a higher value is slower with smaller output.
    """

    # Convert depth map to a uint16 png
    depth_image = (depth_map * 256.0).astype(np.uint16)

    if version == 'cv2':
        ret = cv2.imwrite(save_path, depth_image, [cv2.IMWRITE_PNG_COMPRESSION, png_compression])

        if not ret:
            raise RuntimeError('Could not save depth map')

    elif version == 'pypng':
        with open(save_path, 'wb') as f:
            depth_image = (depth_map * 256.0).astype(np.uint16)
            writer = png.Writer(width=depth_image.shape[1],
                                height=depth_image.shape[0],
                                bitdepth=16,
                                greyscale=True)
            writer.write(f, depth_image)

    else:
        raise ValueError('Invalid version', version)


def get_depth_point_cloud(depth_map, cam_p, min_v=0, flatten=True, in_cam0_frame=True):
    """Calculates the point cloud from a depth map given the camera parameters

    Args:
        depth_map: depth map
        cam_p: camera p matrix
        min_v: amount to crop off the top
        flatten: flatten point cloud to (3, N), otherwise return the point cloud
            in xyz_map (3, H, W) format. (H, W, 3) points can be retrieved using
            xyz_map.transpose(1, 2, 0)
        in_cam0_frame: (optional) If True, shifts the point cloud into cam_0 frame.
            If False, returns the point cloud in the provided camera frame

    Returns:
        point_cloud: (3, N) point cloud
    """

    depth_map_shape = depth_map.shape[0:2]

    if min_v > 0:
        # Crop top part
        depth_map[0:min_v] = 0.0

    xx, yy = np.meshgrid(
        np.linspace(0, depth_map_shape[1] - 1, depth_map_shape[1]),
        np.linspace(0, depth_map_shape[0] - 1, depth_map_shape[0]))

    # Calibration centre x, centre y, focal length
    centre_u = cam_p[0, 2]
    centre_v = cam_p[1, 2]
    focal_length = cam_p[0, 0]

    i = xx - centre_u
    j = yy - centre_v

    # Similar triangles ratio (x/i = d/f)
    ratio = depth_map / focal_length
    x = i * ratio
    y = j * ratio
    z = depth_map

    if in_cam0_frame:
        # Return the points in cam_0 frame
        # Get x offset (b_cam) from calibration: cam_p[0, 3] = (-f_x * b_cam)
        x_offset = -cam_p[0, 3] / focal_length

        # TODO: mask out invalid points
        point_cloud_map = np.asarray([x + x_offset, y, z])

    else:
        # Return the points in the provided camera frame
        point_cloud_map = np.asarray([x, y, z])

    if flatten:
        point_cloud = np.reshape(point_cloud_map, (3, -1))
        return point_cloud.astype(np.float32)
    else:
        return point_cloud_map.astype(np.float32)


def project_depths(point_cloud, cam_p, image_shape, max_depth=100.0):
    """Projects a point cloud into image space and saves depths per pixel.

    Args:
        point_cloud: (3, N) Point cloud in cam0
        cam_p: camera projection matrix
        image_shape: image shape [h, w]
        max_depth: optional, max depth for inversion

    Returns:
        projected_depths: projected depth map
    """

    # Only keep points in front of the camera
    all_points = point_cloud.T

    # Save the depth corresponding to each point
    points_in_img = calib_utils.project_pc_to_image(all_points.T, cam_p)
    points_in_img_int = np.int32(np.round(points_in_img))

    # Remove points outside image
    valid_indices = \
        (points_in_img_int[0] >= 0) & (points_in_img_int[0] < image_shape[1]) & \
        (points_in_img_int[1] >= 0) & (points_in_img_int[1] < image_shape[0])

    all_points = all_points[valid_indices]
    points_in_img_int = points_in_img_int[:, valid_indices]

    # Invert depths
    all_points[:, 2] = max_depth - all_points[:, 2]

    # Only save valid pixels, keep closer points when overlapping
    projected_depths = np.zeros(image_shape)
    valid_indices = [points_in_img_int[1], points_in_img_int[0]]
    projected_depths[valid_indices] = [
        max(projected_depths[
            points_in_img_int[1, idx], points_in_img_int[0, idx]],
            all_points[idx, 2])
        for idx in range(points_in_img_int.shape[1])]

    projected_depths[valid_indices] = \
        max_depth - projected_depths[valid_indices]

    return projected_depths.astype(np.float32)

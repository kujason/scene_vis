import os
import time

import numpy as np
import pykitti
import vtk

from md3d.core import transform_utils
from md3d.datasets.kitti import depth_map_utils, obj_utils
from md3d.datasets.kitti.raw import raw_utils
from md3d.utils import demo_utils
from md3d.visualization.vtk_wrapper import vtk_utils
from md3d.visualization.vtk_wrapper.vtk_boxes import VtkBoxes
from md3d.visualization.vtk_wrapper.vtk_point_cloud import VtkPointCloud
from md3d.visualization.vtk_wrapper.vtk_pyramid_boxes import VtkPyramidBoxes


def np_wrap_to_pi(angles):
    """Wrap angles between [-pi, pi]. Angles right at -pi or pi may flip."""
    return (angles + np.pi) % (2 * np.pi) - np.pi


def get_velo_points(raw_data, frame_idx):
    velo_points = raw_data.get_velo(frame_idx)
    points = velo_points[:, 0:3]
    intensities = velo_points[:, 3]

    return points, intensities


def save_screenshot(vtk_render_window, png_writer,
                    output_dir, sample_name):
    """Saves a screenshot of the current render window
    """
    # Update
    window_to_image_filter = vtk.vtkWindowToImageFilter()
    window_to_image_filter.SetInput(vtk_render_window)
    window_to_image_filter.Update()

    # Take a screenshot and save to file
    file_name = output_dir + "/{}.png".format(sample_name)
    png_writer.SetFileName(file_name)
    png_writer.SetInputData(window_to_image_filter.GetOutput())
    png_writer.Write()


def main():

    ####################
    # Options
    ####################

    # drive_id = '2011_09_26_drive_0001_sync'  # No moving cars, depth completion sample 0 area
    # drive_id = '2011_09_26_drive_0002_sync'  # depth completion sample 0
    # drive_id = '2011_09_26_drive_0005_sync'  # (gps bad)
    # drive_id = '2011_09_26_drive_0005_sync'  # (moving) (gps bad)
    # drive_id = '2011_09_26_drive_0009_sync'  # missing velo?
    # drive_id = '2011_09_26_drive_0011_sync'  # both moving, then traffic light)
    # drive_id = '2011_09_26_drive_0013_sync'  # both moving
    # drive_id = '2011_09_26_drive_0014_sync'  # both moving, sample 104 (314)
    # drive_id = '2011_09_26_drive_0015_sync'  # both moving
    # drive_id = '2011_09_26_drive_0017_sync'  # other only, traffic light, moving across
    # drive_id = '2011_09_26_drive_0018_sync'  # other only, traffic light, forward
    # drive_id = '2011_09_26_drive_0019_sync'  # both moving, some people at end, wonky gps
    # drive_id = '2011_09_26_drive_0020_sync'  # gps drift
    # drive_id = '2011_09_26_drive_0022_sync'  # ego mostly, then both, long
    # drive_id = '2011_09_26_drive_0023_sync'  # sample 169 (474)
    # drive_id = '2011_09_26_drive_0027_sync'  # both moving, fast, straight
    # drive_id = '2011_09_26_drive_0028_sync'  # both moving, opposite
    # drive_id = '2011_09_26_drive_0029_sync'  # both moving, good gps (430)
    # drive_id = '2011_09_26_drive_0032_sync'  # both moving, following some cars
    # drive_id = '2011_09_26_drive_0035_sync'  #
    # drive_id = '2011_09_26_drive_0036_sync'  # (long) behind red truck
    # drive_id = '2011_09_26_drive_0039_sync'  # ok, 1 moving
    drive_id = '2011_09_26_drive_0046_sync'  # (short) only 1 moving at start
    # drive_id = '2011_09_26_drive_0048_sync'  # ok but short, no movement
    # drive_id = '2011_09_26_drive_0052_sync'  #
    # drive_id = '2011_09_26_drive_0056_sync'  #
    # drive_id = '2011_09_26_drive_0057_sync'  # (gps sinking)
    # drive_id = '2011_09_26_drive_0059_sync'  #
    # drive_id = '2011_09_26_drive_0060_sync'  #
    # drive_id = '2011_09_26_drive_0061_sync'  # ego only, ok, long, bumpy, some gps drift
    # drive_id = '2011_09_26_drive_0064_sync'  # Smart car, sample 25
    # drive_id = '2011_09_26_drive_0070_sync'  #
    # drive_id = '2011_09_26_drive_0079_sync'  #
    # drive_id = '2011_09_26_drive_0086_sync'  # (medium) uphill
    # drive_id = '2011_09_26_drive_0087_sync'  #
    # drive_id = '2011_09_26_drive_0091_sync'  #
    # drive_id = '2011_09_26_drive_0093_sync'  # Sample 50 (bad)
    # drive_id = '2011_09_26_drive_0106_sync'  # Two cyclists on right
    # drive_id = '2011_09_26_drive_0113_sync'  #
    # drive_id = '2011_09_26_drive_0117_sync'  # (long)

    # drive_id = '2011_09_28_drive_0002_sync'  # campus
    # drive_id = '2011_09_28_drive_0016_sync'  # campus
    # drive_id = '2011_09_28_drive_0021_sync'  # campus
    # drive_id = '2011_09_28_drive_0034_sync'  #
    # drive_id = '2011_09_28_drive_0037_sync'  #
    # drive_id = '2011_09_28_drive_0038_sync'  #
    # drive_id = '2011_09_28_drive_0039_sync'  # busy campus, bad gps
    # drive_id = '2011_09_28_drive_0043_sync'  #
    # drive_id = '2011_09_28_drive_0045_sync'  #
    # drive_id = '2011_09_28_drive_0179_sync'  #

    # drive_id = '2011_09_29_drive_0026_sync'  #
    # drive_id = '2011_09_29_drive_0071_sync'  #

    # drive_id = '2011_09_30_drive_0020_sync'  #
    # drive_id = '2011_09_30_drive_0027_sync'  # (long)
    # drive_id = '2011_09_30_drive_0028_sync'  # (long) bad gps
    # drive_id = '2011_09_30_drive_0033_sync'  #
    # drive_id = '2011_09_30_drive_0034_sync'  #

    # drive_id = '2011_10_03_drive_0042_sync'  #
    # drive_id = '2011_10_03_drive_0047_sync'  #

    # obj_mapping_path = '../sequencing/samples.txt'
    # obj_mapping = np.loadtxt(obj_mapping_path, np.int32)

    # base_dir = os.path.expanduser('~/Kitti/raw/' + category)
    raw_dir = os.path.expanduser('~/Kitti/raw')

    # max_fps = 10.0

    # vtk_window_size = (1280, 720)
    vtk_window_size = (900, 600)
    # vtk_window_size = (400, 200)

    save_images = False

    point_cloud_source = 'lidar'
    # point_cloud_source = 'fast'
    # point_cloud_source = 'multiscale'
    # point_cloud_source = 'experimental'

    # first_frame_idx = None
    # first_frame_pose = None

    # Load raw data
    drive_date = drive_id[0:10]
    drive_num_str = drive_id[17:21]
    raw_data = pykitti.raw(raw_dir, drive_date, drive_num_str)

    # Check that velo length matches timestamps?
    if len(raw_data.velo_files) != len(raw_data.timestamps):
        raise ValueError('velo files and timestamps have different length!')

    train_mapping = np.loadtxt(
        '/shared/Kitti/object/devkit_object/mapping/train_mapping.txt', np.str)
    train_rand_int = np.loadtxt(
        '/shared/Kitti/object/devkit_object/mapping/train_rand_sep_lines.txt', np.int32) - 1

    obj_labels_dir = os.path.expanduser('~/Kitti/object/training/label_2')

    frame_range = (0, len(raw_data.timestamps))
    # frame_range = (0, 100)
    # frame_range = (0, 10)
    # frame_range = (20, 30)
    # frame_range = (125, 140)
    # frame_range = (0, 150)
    # frame_range = (440, 442)
    # frame_range = (441, 442)
    # frame_range = (440, 452)
    # frame_range = (440, 512)
    # frame_range = (500, 502)
    # frame_range = (500, 512)

    # for frame_idx in range(len(raw_data.timestamps)):
    # for frame_idx in range(457, 459):

    camera_viewpoint = 'front'
    camera_viewpoint = 'elevated'

    # camera_zoom = 2.2
    # camera_viewpoint = (0.0, -5.0, -30.0)
    # camera_fp = (0.0, 1.0, 30.0)

    # viewpoint = 'front'
    # camera_zoom = 0.6
    # camera_pos = (0.0, 0.0, 0.0)
    # camera_fp = (0.0, 0.0, 2000.0)
    # vtk_window_size = (1000, 500)

    # viewpoint = 'bev'
    # camera_zoom = 1.0
    # # camera_pos = (0.0, -15.0, -25.0)
    # # camera_pos = (0.0, 0.0, 0.0)
    # camera_fp = (0.0, 1.0, 30.0)

    ####################
    # End of Options
    ####################

    # Setup output folder

    # drive_name = category + '_' + date + '_' + drive
    # images_out_dir = 'outputs/point_clouds/' + drive_name + '/' + point_cloud_source + '_' + viewpoint
    # os.makedirs(images_out_dir, exist_ok=True)

    # max_loop_time = 1.0 / max_fps

    vtk_renderer = demo_utils.setup_vtk_renderer()

    vtk_render_window = demo_utils.setup_vtk_render_window(
        'Overlaid Point Cloud', vtk_window_size, vtk_renderer)
    # demo_utils.setup_vtk_camera(vtk_renderer, pitch=150.0, zoom=0.5)

    # vtk_renderer, vtk_render_window = setup_vtk_renderer(
    #     vtk_point_cloud, drive_name, window_size=vtk_window_size,
    #     camera_zoom=camera_zoom, camera_pos=camera_pos, camera_fp=camera_fp)
    # png_writer = vtk.vtkPNGWriter()

    vtk_interactor = vtk.vtkRenderWindowInteractor()
    vtk_interactor.SetRenderWindow(vtk_render_window)
    vtk_interactor.SetInteractorStyle(vtk_utils.ToggleActorsInteractorStyle(None, vtk_renderer))
    vtk_interactor.Initialize()

    drive_date_dir = raw_dir + '/{}'.format(drive_date)

    # Setup iterators and paths
    cam2_iterator = raw_data.cam2
    velo_iterator = None
    depth_map_dir = None

    if point_cloud_source == 'lidar':
        velo_iterator = raw_data.velo
    elif point_cloud_source == 'fast':
        depth_map_dir = os.path.join(raw_data.data_path, 'depth_ip_basic_fast')
    elif point_cloud_source == 'multiscale':
        depth_map_dir = os.path.join(raw_data.data_path, 'depth_ip_basic_ms')
    elif point_cloud_source == 'experimental':
        pass
    else:
        raise ValueError('Invalid point cloud source {}'.format(
            point_cloud_source))

    cam_p = raw_data.calib.P_rect_20

    # drive_date_dir = '/home/j3ku/wavelab/datasets/Kitti/raw/2011_10_03'

    # Load poses from matlab
    # poses_path = drive_date_dir + '/{}/poses.mat'.format(drive_id)
    # poses_mat = scipy.io.loadmat(poses_path)
    # imu_ref_poses = np.asarray(poses_mat['pose'][0])

    # Load poses from ORBSLAM2
    poses = np.loadtxt(
        raw_dir + '/orbslam2_cam0_poses/{}/{}_cam0_poses.txt'.format(drive_date, drive_id))
    poses = poses.reshape((-1, 3, 4))
    poses = np.pad(poses, ((0, 0), (0, 1), (0, 0)), mode='constant', constant_values=0)
    poses[:, 3, 3] = 1.0

    # Read calibrations
    # Calculate frame transformations
    cam2cam_calib = raw_utils.get_cam2cam_calib(drive_date_dir)
    imu2velo_calib = raw_utils.get_imu2velo_calib(drive_date_dir)
    velo2cam_calib = raw_utils.get_velo2cam_calib(drive_date_dir)

    # Setup transformation matrix in velo frame
    # r_imu_velo = np.reshape(imu2velo_calib['R'], [3, 3])
    # t_imu_velo = np.reshape(imu2velo_calib['T'], [3, 1])
    # tf_imu_calib_velo_calib = np.hstack([r_imu_velo, t_imu_velo])
    # tf_imu_calib_velo_calib = np.vstack([tf_imu_calib_velo_calib, [0, 0, 0, 1]])

    tf_velo_calib_imu_calib = raw_data.calib.T_velo_imu
    tf_imu_calib_velo_calib = transform_utils.invert_tf(tf_velo_calib_imu_calib)

    # Setup transformation matrix in velo frame
    # r_velo_cam0 = np.reshape(velo2cam_calib['R'], [3, 3])
    # t_velo_cam0 = np.reshape(velo2cam_calib['T'], [3, 1])
    # tf_velo_calib_cam0_calib = np.hstack([r_velo_cam0, t_velo_cam0])
    # tf_velo_calib_cam0_calib = np.vstack([tf_velo_calib_cam0_calib, [0, 0, 0, 1]])

    tf_cam0_calib_velo_calib = raw_data.calib.T_cam0_velo
    tf_velo_calib_cam0_calib = transform_utils.invert_tf(tf_cam0_calib_velo_calib)

    # # Setup cam0 -> cam2 transformation matrix
    # r_cam2_cam0 = cam2cam_calib['R_02'].reshape(3, 3)
    # t_cam2_cam0 = cam2cam_calib['T_02'].reshape(3, 1)
    # tf_cam2_calib_cam0_calib = np.hstack([r_cam2_cam0, t_cam2_cam0])
    # tf_cam2_calib_cam0_calib = np.vstack([tf_cam2_calib_cam0_calib, [0, 0, 0, 1]])

    # Create VtkAxes
    vtk_axes = vtk.vtkAxesActor()
    vtk_axes.SetTotalLength(1, 1, 1)
    vtk_renderer.AddActor(vtk_axes)

    sample_idx = 0
    for frame_idx in range(*frame_range):

        # Point cloud actor wrapper
        vtk_pc = VtkPointCloud()
        vtk_pc.vtk_actor.GetProperty().SetPointSize(2)

        # all_vtk_actors.append(vtk_point_cloud.vtk_actor)
        vtk_renderer.AddActor(vtk_pc.vtk_actor)

        print('{} / {}'.format(frame_idx, len(raw_data.timestamps) - 1))

        # Load next frame data
        load_start_time = time.time()
        # rgb_image = np.asarray(next(cam2_iterator))
        rgb_image = np.asarray(raw_data.get_cam2(frame_idx))
        bgr_image = rgb_image[..., ::-1]

        if point_cloud_source == 'lidar':
            velo_points, velo_intensities = get_velo_points(raw_data, frame_idx)

            # Transform point cloud to cam_0 frame
            velo_curr_points_padded = np.pad(
                velo_points, [[0, 0], [0, 1]],
                constant_values=1.0, mode='constant')

            # R_rect_00 already applied in T_cam0_velo
            # T_cam0_velo = R_rect_00 @ T_cam0_velo_unrect
            cam0_curr_pc_all_padded = raw_data.calib.T_cam0_velo @ velo_curr_points_padded.T

            pass
            # cam0_curr_points = cam0_curr_pc.T

        elif point_cloud_source in ['fast', 'multiscale']:

            # depth_map_path = depth_map_dir + '/{:010d}.png'.format(frame_idx)
            # depth_map = depth_map_utils.read_depth_map(depth_map_path)
            # points_cam2 = depth_map_utils.get_depth_point_cloud(depth_map, cam_p).T
            pass
        else:
            raise ValueError('Invalid point cloud source')

        # print('load\t\t', time.time() - load_start_time)

        # Project velodyne points
        projection_start_time = time.time()
        # points_in_img = calib_utils.project_pc_to_image(points_cam2.T, cam_p)

        if point_cloud_source == 'lidar':
            # calib_utils.lidar_to_cam_frame()
            # points = points_velo[image_mask]

            # points_in_img = calib_utils.project_pc_to_image(cam0_curr_pc_all_padded[0:3], cam_p)
            # points_in_img_int = np.round(points_in_img).astype(np.int32)
            #
            # image_filter = obj_utils.points_in_img_filter(points_in_img_int, bgr_image.shape)

            # image_mask = (points_in_img_int[0] >= 0) & (points_in_img_int[0] < bgr_image.shape[1]) & \
            #              (points_in_img_int[1] >= 0) & (points_in_img_int[1] < bgr_image.shape[0])

            # cam0_curr_pc_padded = cam0_curr_pc_all_padded[:, image_filter]
            # cam0_curr_points = cam0_curr_pc_padded[0:3].T

            # points = points_cam2[image_mask]
            # points_in_img_int_valid = points_in_img_int[:, image_filter]
            # point_colours = bgr_image[points_in_img_int_valid[1], points_in_img_int_valid[0]]

            # R_rect_00 already applied in T_cam0_velo
            cam0_curr_pc_padded = tf_cam0_calib_velo_calib @ velo_curr_points_padded.T
            # cam0_curr_pc_padded = cam0_curr_points.T
            point_colours = np.repeat(np.round(velo_intensities * 255).astype(np.uint8), 3).reshape(-1, 3)
        else:
            cam0_curr_points = points_cam2[image_mask]
            point_colours = bgr_image[points_in_img_int[1], points_in_img_int[0]][image_mask]

        print('projection\t', time.time() - projection_start_time)

        # Get calibration transformations
        tf_velo_calib_imu_calib = transform_utils.invert_tf(tf_imu_calib_velo_calib)
        tf_cam0_calib_imu_calib = tf_cam0_calib_velo_calib @ tf_velo_calib_imu_calib

        cam0_ref_pose = poses[frame_idx]
        tf_cam0_ref_cam0_curr = cam0_ref_pose
        cam0_ref_pc_padded = tf_cam0_ref_cam0_curr @ cam0_curr_pc_padded

        # VtkPointCloud
        vtk_pc_start_time = time.time()
        vtk_pc.set_points(cam0_ref_pc_padded[0:3].T, point_colours)
        print('vtk_pc\t\t', time.time() - vtk_pc_start_time)

        vtk_pc_pose = VtkPointCloud()
        vtk_pc_pose.vtk_actor.GetProperty().SetPointSize(5)
        vtk_pc_pose.set_points(np.reshape(cam0_ref_pose[0:3, 3], [-1, 3]))
        # vtk_renderer.AddActor(vtk_pc_pose.vtk_actor)

        # Render
        render_start_time = time.time()
        # Reset the clipping range to show all points
        vtk_renderer.ResetCameraClippingRange()
        vtk_render_window.Render()
        print('render\t\t', time.time() - render_start_time)

        # Move camera
        if camera_viewpoint == 'front':
            cam0_curr_vtk_cam_position = [0.0, 0.0, 0.0, 1.0]
        elif camera_viewpoint == 'elevated':
            cam0_curr_vtk_cam_position = [0.0, -5.0, -10.0, 1.0]
        elif camera_viewpoint == '':
            pass
        else:
            raise ValueError('Invalid camera_pos', camera_viewpoint)

        cam0_ref_vtk_cam_pos = tf_cam0_ref_cam0_curr @ cam0_curr_vtk_cam_position

        cam0_curr_vtk_cam_fp = [0.0, 0.0, 20.0, 1.0]
        cam0_ref_vtk_cam_fp = tf_cam0_ref_cam0_curr @ cam0_curr_vtk_cam_fp

        current_cam = vtk_renderer.GetActiveCamera()
        vtk_renderer.ResetCamera()
        current_cam.SetViewUp(0, -1, 0)

        current_cam.SetPosition(cam0_ref_vtk_cam_pos[0:3])
        current_cam.SetFocalPoint(*cam0_ref_vtk_cam_fp[0:3])

        current_cam.Zoom(0.5)
        vtk_renderer.ResetCameraClippingRange()
        vtk_renderer.GetRenderWindow().Render()

        print('---')

    print('Done')

    # Keep window open
    vtk_interactor.Start()


if __name__ == '__main__':
    main()

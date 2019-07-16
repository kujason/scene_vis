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

    raw_dir = os.path.expanduser('~/Kitti/raw')

    vtk_window_size = (900, 600)

    save_images = False

    point_cloud_source = 'lidar'

    # Load raw data
    drive_date = drive_id[0:10]
    drive_num_str = drive_id[17:21]
    raw_data = pykitti.raw(raw_dir, drive_date, drive_num_str)

    # Check that velo length matches timestamps?
    if len(raw_data.velo_files) != len(raw_data.timestamps):
        raise ValueError('velo files and timestamps have different length!')

    frame_range = (0, len(raw_data.timestamps))
    # frame_range = (0, 100)

    camera_viewpoint = 'front'
    camera_viewpoint = 'elevated'
    ####################
    # End of Options
    ####################

    vtk_renderer = demo_utils.setup_vtk_renderer()

    vtk_render_window = demo_utils.setup_vtk_render_window(
        'Overlaid Point Cloud', vtk_window_size, vtk_renderer)

    vtk_interactor = vtk.vtkRenderWindowInteractor()
    vtk_interactor.SetRenderWindow(vtk_render_window)
    vtk_interactor.SetInteractorStyle(vtk_utils.ToggleActorsInteractorStyle(None, vtk_renderer))
    vtk_interactor.Initialize()

    if point_cloud_source != 'lidar':
        raise ValueError('Invalid point cloud source {}'.format(
            point_cloud_source))

    # Load poses from ORBSLAM2
    poses = np.loadtxt(
        raw_dir + '/orbslam2_cam0_poses/{}/{}_cam0_poses.txt'.format(drive_date, drive_id))
    poses = poses.reshape((-1, 3, 4))
    poses = np.pad(poses, ((0, 0), (0, 1), (0, 0)), mode='constant', constant_values=0)
    poses[:, 3, 3] = 1.0

    # Read calibrations
    tf_velo_calib_imu_calib = raw_data.calib.T_velo_imu
    tf_imu_calib_velo_calib = transform_utils.invert_tf(tf_velo_calib_imu_calib)

    tf_cam0_calib_velo_calib = raw_data.calib.T_cam0_velo

    # Create VtkAxes
    vtk_axes = vtk.vtkAxesActor()
    vtk_axes.SetTotalLength(1, 1, 1)
    vtk_renderer.AddActor(vtk_axes)

    for frame_idx in range(*frame_range):

        # Point cloud actor wrapper
        vtk_pc = VtkPointCloud()
        vtk_pc.vtk_actor.GetProperty().SetPointSize(2)

        # all_vtk_actors.append(vtk_point_cloud.vtk_actor)
        vtk_renderer.AddActor(vtk_pc.vtk_actor)

        print('{} / {}'.format(frame_idx, len(raw_data.timestamps) - 1))

        # Load next frame data
        load_start_time = time.time()
        rgb_image = np.asarray(raw_data.get_cam2(frame_idx))
        bgr_image = rgb_image[..., ::-1]

        if point_cloud_source == 'lidar':
            velo_points, velo_intensities = get_velo_points(raw_data, frame_idx)

            # Transform point cloud to cam_0 frame
            velo_curr_points_padded = np.pad(
                velo_points, [[0, 0], [0, 1]],
                constant_values=1.0, mode='constant')

        else:
            raise ValueError('Invalid point cloud source')

        print('load\t\t', time.time() - load_start_time)

        # Project velodyne points
        projection_start_time = time.time()

        if point_cloud_source == 'lidar':
            cam0_curr_pc_padded = tf_cam0_calib_velo_calib @ velo_curr_points_padded.T
            point_colours = np.repeat(
                np.round(velo_intensities * 255).astype(np.uint8), 3).reshape(-1, 3)

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

import os
import time

import numpy as np
import pykitti
import vtk

import core
from core import demo_utils, depth_map_utils
from core.visualization.vtk_wrapper import vtk_utils
from core.visualization.vtk_wrapper.vtk_point_cloud_glyph import VtkPointCloudGlyph
from datasets.kitti.obj import obj_utils, calib_utils


def get_velo_points(raw_data, frame_idx):
    velo_points = raw_data.get_velo(frame_idx)

    # Filter points to certain area
    points = velo_points[:, 0:3]
    area_extents = np.asarray([[0, 100], [-50, 50], [-5, 1]], dtype=np.float32)
    area_filter = \
        (points[:, 0] > area_extents[0, 0]) & \
        (points[:, 0] < area_extents[0, 1]) & \
        (points[:, 1] > area_extents[1, 0]) & \
        (points[:, 1] < area_extents[1, 1]) & \
        (points[:, 2] > area_extents[2, 0]) & \
        (points[:, 2] < area_extents[2, 1])
    points = points[area_filter]

    return points


def main():
    ##############################
    # Options
    ##############################
    """Note: Run scripts/depth_completion/save_depth_maps_raw.py first"""

    raw_dir = os.path.expanduser('~/Kitti/raw')
    drive_id = '2011_09_26_drive_0023_sync'

    vtk_window_size = (1280, 720)
    max_fps = 30.0

    # point_cloud_source = 'lidar'
    # fill_type = None

    point_cloud_source = 'depth'
    fill_type = 'multiscale'

    # Load raw data
    drive_date = drive_id[0:10]
    drive_num_str = drive_id[17:21]
    raw_data = pykitti.raw(raw_dir, drive_date, drive_num_str)

    # Check that velo length matches timestamps?
    if len(raw_data.velo_files) != len(raw_data.timestamps):
        raise ValueError('velo files and timestamps have different length!')

    frame_range = (0, len(raw_data.timestamps))

    ##############################

    min_loop_time = 1.0 / max_fps

    vtk_renderer = demo_utils.setup_vtk_renderer()
    vtk_render_window = demo_utils.setup_vtk_render_window(
        'Overlaid Point Cloud', vtk_window_size, vtk_renderer)

    # Setup camera
    current_cam = vtk_renderer.GetActiveCamera()
    current_cam.SetViewUp(0, 0, 1)
    current_cam.SetPosition(0.0, -8.0, -15.0)
    current_cam.SetFocalPoint(0.0, 0.0, 20.0)
    current_cam.Zoom(0.7)

    # Create VtkAxes
    vtk_axes = vtk.vtkAxesActor()
    vtk_axes.SetTotalLength(2, 2, 2)

    # Setup interactor
    vtk_interactor = vtk.vtkRenderWindowInteractor()
    vtk_interactor.SetRenderWindow(vtk_render_window)
    vtk_interactor.SetInteractorStyle(
        vtk_utils.ToggleActorsInteractorStyle(None, vtk_renderer, current_cam, vtk_axes))
    vtk_interactor.Initialize()

    if point_cloud_source not in ['lidar', 'depth']:
        raise ValueError('Invalid point cloud source {}'.format(
            point_cloud_source))

    cam_p = raw_data.calib.P_rect_20

    # Point cloud
    vtk_pc = VtkPointCloudGlyph()

    # Add actors
    vtk_renderer.AddActor(vtk_axes)
    vtk_renderer.AddActor(vtk_pc.vtk_actor)

    for frame_idx in range(*frame_range):

        loop_start_time = time.time()

        print('{} / {}'.format(frame_idx, len(raw_data.timestamps) - 1))

        # Load next frame data
        load_start_time = time.time()

        rgb_image = np.asarray(raw_data.get_cam2(frame_idx))
        bgr_image = rgb_image[..., ::-1]

        if point_cloud_source == 'lidar':
            velo_points = get_velo_points(raw_data, frame_idx)

            # Transform point cloud to cam_0 frame
            velo_curr_points_padded = np.pad(
                velo_points, [[0, 0], [0, 1]],
                constant_values=1.0, mode='constant')

            cam0_curr_pc_all_padded = raw_data.calib.T_cam0_velo @ velo_curr_points_padded.T

            # Project velodyne points
            projection_start_time = time.time()
            points_in_img = calib_utils.project_pc_to_image(cam0_curr_pc_all_padded[0:3], cam_p)
            points_in_img_int = np.round(points_in_img).astype(np.int32)
            print('projection\t', time.time() - projection_start_time)

            image_filter = obj_utils.points_in_img_filter(points_in_img_int, bgr_image.shape)

            cam0_curr_pc = cam0_curr_pc_all_padded[0:3, image_filter]

            points_in_img_int_valid = points_in_img_int[:, image_filter]
            point_colours = bgr_image[points_in_img_int_valid[1], points_in_img_int_valid[0]]

        elif point_cloud_source == 'depth':
            depth_maps_dir = core.data_dir() + \
                '/depth_completion/raw/{}/depth_02_{}'.format(drive_id, fill_type)
            depth_map_path = depth_maps_dir + '/{:010d}.png'.format(frame_idx)
            depth_map = depth_map_utils.read_depth_map(depth_map_path)
            cam0_curr_pc = depth_map_utils.get_depth_point_cloud(depth_map, cam_p)

            point_colours = bgr_image.reshape(-1, 3)

            # Mask to valid points
            valid_mask = (cam0_curr_pc[2] != 0)
            cam0_curr_pc = cam0_curr_pc[:, valid_mask]
            point_colours = point_colours[valid_mask]
        else:
            raise ValueError('Invalid point cloud source')

        print('load\t\t', time.time() - load_start_time)

        # VtkPointCloud
        vtk_pc_start_time = time.time()
        vtk_pc.set_points(cam0_curr_pc.T, point_colours)
        print('vtk_pc\t\t', time.time() - vtk_pc_start_time)

        # Reset the clipping range to show all points
        vtk_renderer.ResetCameraClippingRange()

        # Render
        render_start_time = time.time()
        vtk_render_window.Render()
        print('render\t\t', time.time() - render_start_time)

        # Pause to keep frame rate under max
        loop_run_time = time.time() - loop_start_time
        print('loop\t\t', loop_run_time)
        if loop_run_time < min_loop_time:
            time.sleep(min_loop_time - loop_run_time)

        print('---')

    print('Done')

    # Keep window open
    vtk_interactor.Start()


if __name__ == '__main__':
    main()

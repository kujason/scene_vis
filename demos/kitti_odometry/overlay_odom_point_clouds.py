import os
import time

import numpy as np
import pykitti
import vtk

from core import transform_utils, demo_utils
from core.visualization.vtk_wrapper import vtk_utils
from core.visualization.vtk_wrapper.vtk_point_cloud import VtkPointCloud
from datasets.kitti.obj import obj_utils, calib_utils


def get_velo_points(odom_dataset, frame_idx):
    velo_points = odom_dataset.get_velo(frame_idx)

    # Filter points to certain area
    points = velo_points[:, 0:3]
    area_extents = np.array([[0, 100], [-50, 50], [-5, 1]], dtype=np.float32)
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

    ####################
    # Options
    ####################

    odom_dir = os.path.expanduser('~/Kitti/odometry/dataset')

    sequence = '03'

    # max_fps = 10.0

    # vtk_window_size = (1280, 720)
    vtk_window_size = (960, 540)

    save_images = False

    point_cloud_source = 'lidar'
    # point_cloud_source = 'fast'
    # point_cloud_source = 'multiscale'

    # first_frame_idx = None
    # first_frame_pose = None

    # Setup odometry dataset handler
    odom_dataset = pykitti.odometry(odom_dir, sequence)

    # # Check that velo length matches timestamps?
    # if len(odom_dataset.velo_files) != len(odom_dataset.timestamps):
    #     raise ValueError('velo files and timestamps have different length!')

    frame_range = (0, len(odom_dataset.timestamps))
    # frame_range = (0, 100)
    # frame_range = (1, 10)
    # frame_range = (20, 30)
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

    vtk_interactor = vtk.vtkRenderWindowInteractor()
    vtk_interactor.SetRenderWindow(vtk_render_window)
    vtk_interactor.SetInteractorStyle(vtk_utils.ToggleActorsInteractorStyle(None, vtk_renderer))
    vtk_interactor.Initialize()

    cam_p2 = odom_dataset.calib.P_rect_20

    # Load poses
    cam0_poses = odom_dataset.poses

    # Setup camera
    if camera_viewpoint == 'front':
        cam0_curr_vtk_cam_pos = [0.0, 0.0, 0.0, 1.0]
        cam0_curr_vtk_focal_point = [0.0, 0.0, 20.0, 1.0]
    elif camera_viewpoint == 'elevated':
        cam0_curr_vtk_cam_pos = [0.0, -5.0, -15.0, 1.0]
        cam0_curr_vtk_focal_point = [0.0, 0.0, 20.0, 1.0]
    else:
        raise ValueError('Invalid camera_pos', camera_viewpoint)

    # Create VtkAxes
    vtk_axes = vtk.vtkAxesActor()
    vtk_axes.SetTotalLength(2, 2, 2)
    vtk_renderer.AddActor(vtk_axes)

    for frame_idx in range(*frame_range):

        # Point cloud actor wrapper
        vtk_pc = VtkPointCloud()
        vtk_pc.vtk_actor.GetProperty().SetPointSize(2)

        # all_vtk_actors.append(vtk_point_cloud.vtk_actor)
        vtk_renderer.AddActor(vtk_pc.vtk_actor)

        print('{} / {}'.format(frame_idx, len(odom_dataset.timestamps) - 1))

        # Load next frame data
        load_start_time = time.time()
        rgb_image = np.asarray(odom_dataset.get_cam2(frame_idx))
        bgr_image = rgb_image[..., ::-1]

        if point_cloud_source == 'lidar':
            velo_points = get_velo_points(odom_dataset, frame_idx)

            # Transform point cloud to cam_0_curr frame
            velo_curr_points_padded = np.pad(
                velo_points, [[0, 0], [0, 1]],
                constant_values=1.0, mode='constant')
            cam0_curr_pc_all_padded = odom_dataset.calib.T_cam0_velo @ velo_curr_points_padded.T

        elif point_cloud_source == 'multiscale':

            # depth_map_path = depth_map_dir + '/{:010d}.png'.format(frame_idx)
            # depth_map = depth_map_utils.read_depth_map(depth_map_path)
            # points_cam2 = depth_map_utils.get_depth_point_cloud(depth_map, cam_p).T
            raise NotImplementedError()

        else:
            raise ValueError('Invalid point cloud source')
        print('load\t\t', time.time() - load_start_time)

        # Project velodyne points
        projection_start_time = time.time()

        if point_cloud_source == 'lidar':

            # Project into image2
            points_in_img2 = calib_utils.project_pc_to_image(cam0_curr_pc_all_padded[0:3], cam_p2)
            points_in_img2_int = np.round(points_in_img2).astype(np.int32)

            image_filter = obj_utils.points_in_img_filter(points_in_img2_int, bgr_image.shape)

            cam0_curr_pc_padded = cam0_curr_pc_all_padded[:, image_filter]

            points_in_img_int_valid = points_in_img2_int[:, image_filter]
            point_colours = bgr_image[points_in_img_int_valid[1], points_in_img_int_valid[0]]
        else:
            raise ValueError('Invalid point_cloud_source', point_cloud_source)

        print('projection\t', time.time() - projection_start_time)

        # Get pose
        cam0_ref_pose = cam0_poses[frame_idx]
        tf_cam0_ref_cam0_curr = cam0_ref_pose

        # print('cam0_ref_pose\n', np.round(cam0_ref_pose, 3))

        cam0_ref_pc_padded = tf_cam0_ref_cam0_curr @ cam0_curr_pc_padded

        # VtkPointCloud
        vtk_pc_start_time = time.time()
        vtk_pc.set_points(cam0_ref_pc_padded[0:3].T, point_colours)
        print('vtk_pc\t\t', time.time() - vtk_pc_start_time)

        # Display pose
        vtk_pc_pose = VtkPointCloud()
        vtk_pc_pose.vtk_actor.GetProperty().SetPointSize(5)
        vtk_pc_pose.set_points(np.reshape(cam0_ref_pose[0:3, 3], [-1, 3]))
        vtk_renderer.AddActor(vtk_pc_pose.vtk_actor)

        cam0_ref_vtk_cam_pos = tf_cam0_ref_cam0_curr.dot(cam0_curr_vtk_cam_pos)
        cam0_ref_vtk_focal_point = tf_cam0_ref_cam0_curr.dot(cam0_curr_vtk_focal_point)

        current_cam = vtk_renderer.GetActiveCamera()
        vtk_renderer.ResetCamera()
        current_cam.SetViewUp(0, -1, 0)
        current_cam.SetPosition(cam0_ref_vtk_cam_pos[0:3])
        current_cam.SetFocalPoint(*cam0_ref_vtk_focal_point[0:3])
        current_cam.Zoom(0.5)

        vtk_renderer.ResetCameraClippingRange()

        # Render
        render_start_time = time.time()
        vtk_render_window.Render()
        print('render\t\t', time.time() - render_start_time)

        print('---')

    print('Done')

    # Keep window open
    vtk_interactor.Start()


if __name__ == '__main__':
    main()

import os
import time

import cv2
import numpy as np
import pykitti
import vtk

import core
from core import demo_utils
from core.visualization.vtk_wrapper import vtk_utils
from core.visualization.vtk_wrapper.vtk_point_cloud_glyph import VtkPointCloudGlyph
from datasets.kitti.obj import obj_utils, calib_utils


def get_velo_points(odom_dataset, frame_idx):
    velo_points = odom_dataset.get_velo(frame_idx)

    # Filter points to certain area
    velo_points[:, 3] = 1
    velo_points = velo_points[velo_points[:, 0] > 0]

    return velo_points


def main():
    """Comparison of ground truth and ORBSLAM2 poses
    https://github.com/raulmur/ORB_SLAM2
    """

    ####################
    # Options
    ####################

    odom_dataset_dir = os.path.expanduser('~/Kitti/odometry/dataset')

    # sequence = '02'
    # sequence = '03'
    # sequence = '08'
    sequence = '09'
    # sequence = '10'
    # sequence = '11'
    # sequence = '12'

    # vtk_window_size = (1280, 720)
    vtk_window_size = (960, 540)
    # vtk_window_size = (400, 300)

    point_cloud_source = 'lidar'
    # point_cloud_source = 'fast'
    # point_cloud_source = 'multiscale'

    # first_frame_idx = None
    # first_frame_pose = None

    # Setup odometry dataset handler
    odom_dataset = pykitti.odometry(odom_dataset_dir, sequence)

    # # Check that velo length matches timestamps?
    # if len(odom_dataset.velo_files) != len(odom_dataset.timestamps):
    #     raise ValueError('velo files and timestamps have different length!')

    frame_range = (0, len(odom_dataset.timestamps))
    # frame_range = (0, 100)

    # camera_viewpoint = 'front'
    camera_viewpoint = 'elevated'
    # camera_viewpoint = 'bev'

    buffer_size = 50
    buffer_update = 10

    save_screenshots = False

    ##############################

    vtk_renderer = demo_utils.setup_vtk_renderer()

    vtk_render_window = demo_utils.setup_vtk_render_window(
        'Overlaid Point Cloud', vtk_window_size, vtk_renderer)

    if save_screenshots:
        vtk_win_to_img_filter, vtk_png_writer = vtk_utils.setup_screenshots(vtk_render_window)

    vtk_interactor = vtk.vtkRenderWindowInteractor()
    vtk_interactor.SetRenderWindow(vtk_render_window)
    vtk_interactor.SetInteractorStyle(vtk_utils.ToggleActorsInteractorStyle(None, vtk_renderer))
    vtk_interactor.Initialize()

    cam_p2 = odom_dataset.calib.P_rect_20

    # Load poses
    cam0_ref_poses_orbslam = np.loadtxt(odom_dataset_dir + '/poses_orbslam2/{}.txt'.format(sequence))
    cam0_ref_poses_orbslam = np.pad(cam0_ref_poses_orbslam, ((0, 0), (0, 4)), mode='constant')
    cam0_ref_poses_orbslam[:, -1] = 1
    cam0_ref_poses_orbslam = cam0_ref_poses_orbslam.reshape((-1, 4, 4))

    cam0_ref_poses_gt = np.asarray(odom_dataset.poses, np.float32)

    # Setup camera
    if camera_viewpoint == 'front':
        cam0_curr_vtk_cam_pos = [0.0, 0.0, 0.0, 1.0]
        cam0_curr_vtk_focal_point = [0.0, 0.0, 20.0, 1.0]
    elif camera_viewpoint == 'elevated':
        cam0_curr_vtk_cam_pos = [0.0, -5.0, -15.0, 1.0]
        cam0_curr_vtk_focal_point = [0.0, 0.0, 20.0, 1.0]
    elif camera_viewpoint == 'bev':
        # camera_zoom = 1.0
        cam0_curr_vtk_cam_pos = [0.0, -50.0, 10.0, 1.0]
        # camera_pos = (0.0, 0.0, 0.0)
        cam0_curr_vtk_focal_point = [0.0, 0.0, 15.0, 1.0]
    else:
        raise ValueError('Invalid camera_pos', camera_viewpoint)

    # Create VtkAxes
    vtk_axes = vtk.vtkAxesActor()
    vtk_axes.SetTotalLength(2, 2, 2)
    vtk_renderer.AddActor(vtk_axes)

    vtk_pc_poses_orbslam = VtkPointCloudGlyph()
    vtk_pc_poses_gt = VtkPointCloudGlyph()

    vtk_pc_poses_orbslam.vtk_actor.GetProperty().SetPointSize(5)
    vtk_pc_poses_gt.vtk_actor.GetProperty().SetPointSize(5)

    actor_buffer = []

    for frame_idx in range(*frame_range):

        print('{} / {}'.format(frame_idx, len(odom_dataset.timestamps) - 1))

        # Load next frame data
        load_start_time = time.time()

        rgb_load_start = time.time()
        bgr_image = cv2.imread(odom_dataset.cam2_files[frame_idx])
        print('rgb_load', time.time() - rgb_load_start)

        if point_cloud_source == 'lidar':

            velo_curr_points_padded = get_velo_points(odom_dataset, frame_idx)

            # Velo in cam0_curr
            cam0_curr_pc_all_padded = odom_dataset.calib.T_cam0_velo @ velo_curr_points_padded.T

            pass
            # cam0_curr_points = cam0_curr_pc.T

        elif point_cloud_source in ['fast', 'multiscale']:

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
            points_in_img2 = calib_utils.project_pc_to_image2(cam0_curr_pc_all_padded, cam_p2)
            points_in_img2_int = np.round(points_in_img2).astype(np.int32)

            image_filter = obj_utils.points_in_img_filter(points_in_img2_int, bgr_image.shape)

            cam0_curr_pc_padded = cam0_curr_pc_all_padded[:, image_filter]

            # points = points_cam2[image_mask]
            points_in_img_int_valid = points_in_img2_int[:, image_filter]
            point_colours = bgr_image[points_in_img_int_valid[1], points_in_img_int_valid[0]]
        else:
            raise ValueError('Invalid point_cloud_source', point_cloud_source)

        print('projection\t', time.time() - projection_start_time)

        # Get pose
        cam0_ref_pose_orbslam = cam0_ref_poses_orbslam[frame_idx]
        cam0_ref_pose_gt = cam0_ref_poses_gt[frame_idx]

        tf_cam0_ref_cam0_curr = cam0_ref_pose_orbslam
        cam0_ref_pc_padded = tf_cam0_ref_cam0_curr @ cam0_curr_pc_padded

        # VtkPointCloud
        vtk_pc = VtkPointCloudGlyph()
        vtk_pc.vtk_actor.GetProperty().SetPointSize(2)

        vtk_pc_start_time = time.time()
        vtk_pc.set_points(cam0_ref_pc_padded[0:3].T, point_colours)
        print('vtk_pc\t\t', time.time() - vtk_pc_start_time)

        # Display orbslam pose
        vtk_pc_poses_orbslam.set_points(
            cam0_ref_poses_orbslam[max(0, frame_idx - buffer_size):frame_idx + 1, 0:3, 3],
            np.tile([255, 0, 0], [buffer_size, 1]))

        # Display gt pose
        vtk_pc_poses_gt.vtk_actor.GetProperty().SetPointSize(5)
        vtk_pc_poses_gt.set_points(
            cam0_ref_poses_gt[max(0, frame_idx - buffer_size):frame_idx + 1, 0:3, 3],
            np.tile([0, 255, 0], [buffer_size, 1]))

        # Add vtk actors
        vtk_renderer.AddActor(vtk_pc.vtk_actor)
        vtk_renderer.AddActor(vtk_pc_poses_orbslam.vtk_actor)
        vtk_renderer.AddActor(vtk_pc_poses_gt.vtk_actor)

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

        actor_buffer.append(vtk_pc.vtk_actor)
        if len(actor_buffer) > buffer_size:
            if frame_idx % buffer_update != 0:
                actor_buffer[frame_idx - buffer_size].SetVisibility(0)

        if save_screenshots:
            screenshot_start_time = time.time()
            screenshot_path = core.data_dir() + '/temp/{:010d}.png'.format(frame_idx)
            vtk_utils.save_screenshot(
                screenshot_path, vtk_win_to_img_filter, vtk_png_writer)
            print('screenshot\t', time.time() - screenshot_start_time)

        print('---')

    for vtk_actor in actor_buffer:
        vtk_actor.SetVisibility(1)

    print('Done')

    # Keep window open
    vtk_interactor.Start()


if __name__ == '__main__':
    main()

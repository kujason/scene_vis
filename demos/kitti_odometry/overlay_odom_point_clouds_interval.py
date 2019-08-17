import os
import time

import numpy as np
import pykitti
import vtk

from core import transform_utils, demo_utils
from core.visualization.vtk_wrapper import vtk_utils
from core.visualization.vtk_wrapper.vtk_point_cloud import VtkPointCloud
from core.visualization.vtk_wrapper.vtk_point_cloud_glyph import VtkPointCloudGlyph
from datasets.kitti.obj import obj_utils, calib_utils


def get_velo_points(odom_dataset, frame_idx):
    velo_points = odom_dataset.get_velo(frame_idx)

    # Filter points to certain area
    points = velo_points[:, 0:3]
    points = points[points[:, 0] > 0]

    return points


def main():

    ####################
    # Options
    ####################

    odom_dir = os.path.expanduser('~/Kitti/odometry/dataset')

    # sequence = '01'
    # sequence = '02'
    # sequence = '03'
    # sequence = '04'
    # sequence = '05'
    # sequence = '06'
    # sequence = '08'
    sequence = '09'

    # max_fps = 10.0

    # vtk_window_size = (1280, 720)
    vtk_window_size = (960, 540)

    save_images = False

    point_cloud_source = 'lidar'
    # point_cloud_source = 'fast'
    # point_cloud_source = 'multiscale'

    # Setup odometry dataset handler
    odom_dataset = pykitti.odometry(odom_dir, sequence)

    # # Check that velo length matches timestamps?
    # if len(odom_dataset.velo_files) != len(odom_dataset.timestamps):
    #     raise ValueError('velo files and timestamps have different length!')

    frame_range = (0, len(odom_dataset.timestamps))
    # frame_range = (0, 200)

    camera_viewpoint = 'front'
    camera_viewpoint = 'elevated'

    render_interval = 4

    actor_buffer = []

    ####################
    # End of Options
    ####################

    vtk_renderer = demo_utils.setup_vtk_renderer()

    vtk_render_window = demo_utils.setup_vtk_render_window(
        'Sequence {}'.format(sequence), vtk_window_size, vtk_renderer)

    vtk_interactor = vtk.vtkRenderWindowInteractor()
    vtk_interactor.SetRenderWindow(vtk_render_window)
    vtk_interactor.SetInteractorStyle(vtk_utils.ToggleActorsInteractorStyle(None, vtk_renderer))
    vtk_interactor.Initialize()

    cam_p2 = odom_dataset.calib.P_rect_20

    # Load poses
    cam0_poses = odom_dataset.poses

    # tf_cam0_calib_velo_calib = odom_dataset.calib.T_cam0_velo
    # tf_velo_calib_cam0_calib = transform_utils.invert_tf(tf_cam0_calib_velo_calib)

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
    vtk_axes.SetTotalLength(1, 1, 1)
    vtk_renderer.AddActor(vtk_axes)

    all_render_times = []

    buffer_points = []
    buffer_colours = []

    for frame_idx in range(*frame_range):

        print('{} / {}'.format(frame_idx, len(odom_dataset.timestamps) - 1))

        # Load next frame data
        load_start_time = time.time()
        rgb_image = np.asarray(odom_dataset.get_cam2(frame_idx))
        bgr_image = rgb_image[..., ::-1]

        if point_cloud_source == 'lidar':
            velo_points = get_velo_points(odom_dataset, frame_idx)

            # Transform point cloud to cam_0 frame
            velo_curr_points_padded = np.pad(
                velo_points, [[0, 0], [0, 1]],
                constant_values=1.0, mode='constant')

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
        # points_in_img = calib_utils.project_pc_to_image(points_cam2.T, cam_p)

        if point_cloud_source == 'lidar':

            # Project into image2
            points_in_img2 = calib_utils.project_pc_to_image(cam0_curr_pc_all_padded[0:3], cam_p2)
            points_in_img2_int = np.round(points_in_img2).astype(np.int32)

            image_filter = obj_utils.points_in_img_filter(points_in_img2_int, bgr_image.shape)

            # image_mask = (points_in_img_int[0] >= 0) & (points_in_img_int[0] < bgr_image.shape[1]) & \
            #              (points_in_img_int[1] >= 0) & (points_in_img_int[1] < bgr_image.shape[0])

            cam0_curr_pc_padded = cam0_curr_pc_all_padded[:, image_filter]

            # cam0_curr_points = cam0_curr_pc_padded[0:3].T

            # points = points_cam2[image_mask]
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

        cam0_ref_points = cam0_ref_pc_padded[0:3].T

        if render_interval > 1:
            buffer_points.extend(cam0_ref_points)
            buffer_colours.extend(point_colours)

        if frame_idx % render_interval == 0 or frame_idx == (frame_range[1] - 1):
            vtk_pc = VtkPointCloudGlyph()
            vtk_pc.vtk_actor.GetProperty().SetPointSize(2)
            vtk_renderer.AddActor(vtk_pc.vtk_actor)
            vtk_pc_start_time = time.time()

            if render_interval > 1:
                vtk_pc.set_points(buffer_points, buffer_colours)
                buffer_points = []
                buffer_colours = []
            else:
                vtk_pc.set_points(cam0_ref_points, point_colours)

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
            vtk_renderer.GetRenderWindow().Render()
            print('render\t\t', time.time() - render_start_time)

            all_render_times.append(time.time() - render_start_time)

        print('---')

        actor_buffer.append(vtk_pc.vtk_actor)
        if len(actor_buffer) > 50:
            if frame_idx % 10 != 0:
                actor_buffer[frame_idx - 50].SetVisibility(0)

    for vtk_actor in actor_buffer:
        vtk_actor.SetVisibility(1)

    print('Done')

    import matplotlib.pyplot as plt
    plt.plot(all_render_times)
    plt.show(block=True)

    # Keep window open
    vtk_interactor.Start()


if __name__ == '__main__':
    main()

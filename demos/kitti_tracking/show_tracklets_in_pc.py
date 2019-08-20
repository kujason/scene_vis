import os
import time

import numpy as np
import pykitti
import scipy.io
import vtk

from core import transform_utils, demo_utils
from core.visualization.vtk_wrapper import vtk_utils
from core.visualization.vtk_wrapper.vtk_boxes import VtkBoxes
from core.visualization.vtk_wrapper.vtk_frustums import VtkFrustums
from core.visualization.vtk_wrapper.vtk_point_cloud_glyph import VtkPointCloudGlyph
from core.visualization.vtk_wrapper.vtk_text_labels import VtkTextLabels
from datasets.kitti.obj import obj_utils, calib_utils
from datasets.kitti.tracking import tracking_utils


def get_velo_points(tracking_data, frame_idx):
    velo_points = tracking_data.get_velo(frame_idx)

    # Filter points to certain area
    points = velo_points[:, 0:3]
    area_extents = np.array([[0, 100], [-50, 50], [-5, 1]],
                            dtype=np.float32)
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

    sequence_id = '0013'
    # sequence_id = '0014'
    # sequence_id = '0015'

    # raw_dir = os.path.expanduser('~/Kitti/raw')
    tracking_dir = os.path.expanduser('~/Kitti/tracking/training')

    tracking_data = pykitti.tracking(tracking_dir, sequence_id)

    max_fps = 20.0
    vtk_window_size = (1280, 720)
    show_pose = True
    point_cloud_source = 'lidar'
    buffer_size = 5

    oxts_path = tracking_data.base_path + '/oxts/{}.txt'.format(sequence_id)
    oxts_data = pykitti.utils.load_oxts_packets_and_poses([oxts_path])
    poses = np.asarray([oxt.T_w_imu for oxt in oxts_data])

    calib_path = tracking_data.base_path + '/calib/{}.txt'.format(sequence_id)
    calib_data = tracking_utils.load_calib(calib_path)

    camera_viewpoint = 'front'
    camera_viewpoint = 'elevated'

    ##############################

    label_path = tracking_data.base_path + '/label_02/{}.txt'.format(sequence_id)
    gt_tracklets = tracking_utils.get_tracklets(label_path)

    min_loop_time = 1.0 / max_fps

    vtk_renderer = demo_utils.setup_vtk_renderer()
    vtk_render_window = demo_utils.setup_vtk_render_window(
        'KITTI Tracking Demo', vtk_window_size, vtk_renderer)

    vtk_interactor = vtk.vtkRenderWindowInteractor()
    vtk_interactor.SetRenderWindow(vtk_render_window)
    vtk_interactor.SetInteractorStyle(vtk_utils.ToggleActorsInteractorStyle(None, vtk_renderer))
    vtk_interactor.Initialize()

    if point_cloud_source != 'lidar':
        raise ValueError('Invalid point cloud source {}'.format(
            point_cloud_source))

    # cam_p = raw_data.calib.P_rect_20
    cam_p = calib_data['P2'].reshape(3, 4)
    r_rect = calib_data['R_rect'].reshape(3, 3)

    # Read calibrations
    tf_velo_calib_imu_calib = calib_data['Tr_imu_velo'].reshape(3, 4)
    tf_velo_calib_imu_calib = np.vstack([tf_velo_calib_imu_calib, np.asarray([0, 0, 0, 1])])
    # tf_velo_calib_imu_calib = raw_data.calib.T_velo_imu
    tf_imu_calib_velo_calib = transform_utils.invert_tf(tf_velo_calib_imu_calib)

    tf_cam0_calib_velo_calib = r_rect @ calib_data['Tr_velo_cam'].reshape(3, 4)
    tf_cam0_calib_velo_calib = np.vstack([tf_cam0_calib_velo_calib, np.asarray([0, 0, 0, 1])])
    # tf_cam0_calib_velo_calib = raw_data.calib.T_cam0_velo
    tf_velo_calib_cam0_calib = transform_utils.invert_tf(tf_cam0_calib_velo_calib)

    # Create VtkAxes
    vtk_axes = vtk.vtkAxesActor()
    vtk_axes.SetTotalLength(2, 2, 2)
    vtk_renderer.AddActor(vtk_axes)

    # Create vtk actors
    vtk_pc_list = []
    for i in range(buffer_size):
        # Point cloud actor wrapper
        vtk_pc = VtkPointCloudGlyph()
        vtk_pc.vtk_actor.GetProperty().SetPointSize(2)

        # all_vtk_actors.append(vtk_point_cloud.vtk_actor)
        vtk_renderer.AddActor(vtk_pc.vtk_actor)
        vtk_pc_list.append(vtk_pc)

    curr_actor_idx = -1

    vtk_boxes = VtkBoxes()
    vtk_text_labels = VtkTextLabels()
    vtk_renderer.AddActor(vtk_boxes.vtk_actor)
    vtk_renderer.AddActor(vtk_text_labels.vtk_actor)

    for frame_idx in range(len(poses)):

        loop_start_time = time.time()

        curr_actor_idx = (curr_actor_idx + 1) % buffer_size
        vtk_pc = vtk_pc_list[curr_actor_idx]

        for i in range(buffer_size):

            if i == curr_actor_idx:
                vtk_pc_list[i].vtk_actor.GetProperty().SetPointSize(3)
            else:
                vtk_pc_list[i].vtk_actor.GetProperty().SetPointSize(0.5)

        print('{} / {}'.format(frame_idx, len(tracking_data.velo_files) - 1))

        # Load next frame data
        load_start_time = time.time()
        # rgb_image = np.asarray(next(cam2_iterator))
        rgb_image = np.asarray(tracking_data.get_cam2(frame_idx))
        bgr_image = rgb_image[..., ::-1]

        if point_cloud_source == 'lidar':
            velo_points = get_velo_points(tracking_data, frame_idx)

            # Transform point cloud to cam_0 frame
            velo_curr_points_padded = np.pad(
                velo_points, [[0, 0], [0, 1]],
                constant_values=1.0, mode='constant')

            cam0_curr_pc_all_padded = tf_cam0_calib_velo_calib @ velo_curr_points_padded.T
            # cam0_curr_pc_all_padded = raw_data.calib.T_cam0_velo @ velo_curr_points_padded.T

        else:
            raise ValueError('Invalid point cloud source')

        print('load\t\t', time.time() - load_start_time)

        # Project velodyne points
        projection_start_time = time.time()

        if point_cloud_source == 'lidar':

            points_in_img = calib_utils.project_pc_to_image(cam0_curr_pc_all_padded[0:3], cam_p)
            points_in_img_int = np.round(points_in_img).astype(np.int32)

            image_filter = obj_utils.points_in_img_filter(points_in_img_int, bgr_image.shape)

            cam0_curr_pc_padded = cam0_curr_pc_all_padded[:, image_filter]

            points_in_img_int_valid = points_in_img_int[:, image_filter]
            point_colours = bgr_image[points_in_img_int_valid[1], points_in_img_int_valid[0]]
        else:
            raise ValueError('Invalid point_cloud_source', point_cloud_source)

        print('projection\t', time.time() - projection_start_time)

        # Get calibration transformations
        tf_velo_calib_imu_calib = transform_utils.invert_tf(tf_imu_calib_velo_calib)
        tf_cam0_calib_imu_calib = tf_cam0_calib_velo_calib @ tf_velo_calib_imu_calib

        tf_imu_calib_cam0_calib = transform_utils.invert_tf(tf_cam0_calib_imu_calib)

        # Get poses
        imu_ref_pose = poses[frame_idx]
        tf_imu_ref_imu_curr = imu_ref_pose

        velo_curr_pc_padded = tf_velo_calib_cam0_calib @ cam0_curr_pc_padded
        imu_curr_pc_padded = tf_imu_calib_velo_calib @ velo_curr_pc_padded
        imu_ref_pc_padded = tf_imu_ref_imu_curr @ imu_curr_pc_padded

        # TODO: Show points in correct frame
        # VtkPointCloud
        vtk_pc_start_time = time.time()
        vtk_pc.set_points(cam0_curr_pc_padded[0:3].T, point_colours)
        # vtk_pc.set_points(imu_ref_pc_padded[0:3].T, point_colours)
        print('vtk_pc\t\t', time.time() - vtk_pc_start_time)

        # if show_pose:
        #     vtk_pc_pose = VtkPointCloudGlyph()
        #     vtk_pc_pose.vtk_actor.GetProperty().SetPointSize(5)
        #     vtk_pc_pose.set_points(np.reshape(imu_ref_pose[0:3, 3], [-1, 3]))
        #     vtk_renderer.AddActor(vtk_pc_pose.vtk_actor)

        # Tracklets
        frame_mask = [tracklet.frame == frame_idx for tracklet in gt_tracklets]
        tracklets_for_frame = gt_tracklets[frame_mask]

        obj_labels = [tracking_utils.tracklet_to_obj_label(tracklet)
                      for tracklet in tracklets_for_frame]
        vtk_boxes.set_objects(
            obj_labels, colour_scheme=vtk_utils.COLOUR_SCHEME_KITTI)

        positions_3d = [obj_label.t for obj_label in obj_labels]
        text_labels = [str(tracklet.id) for tracklet in tracklets_for_frame]
        vtk_text_labels.set_text_labels(positions_3d, text_labels)

        # Move camera
        if camera_viewpoint == 'front':
            imu_curr_cam0_position = tf_imu_calib_cam0_calib @ [0.0, 0.0, 0.0, 1.0]
        elif camera_viewpoint == 'elevated':
            imu_curr_cam0_position = tf_imu_calib_cam0_calib.dot([0.0, -5.0, -10.0, 1.0])
        else:
            raise ValueError('Invalid camera_pos', camera_viewpoint)

        # imu_ref_cam0_position = tf_imu_ref_imu_curr.dot(imu_curr_cam0_position)
        # imu_curr_focal_point = tf_imu_calib_cam0_calib.dot([0.0, 0.0, 20.0, 1.0])
        # imu_ref_focal_point = tf_imu_ref_imu_curr.dot(imu_curr_focal_point)

        current_cam = vtk_renderer.GetActiveCamera()
        vtk_renderer.ResetCamera()
        current_cam.SetViewUp(0, 0, 1)

        current_cam.SetPosition(0.0, -8.0, -20.0)
        # current_cam.SetPosition(imu_ref_cam0_position[0:3])
        current_cam.SetFocalPoint(0.0, 0.0, 20.0)
        # current_cam.SetFocalPoint(*imu_ref_focal_point[0:3])

        # current_cam.Zoom(0.5)

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

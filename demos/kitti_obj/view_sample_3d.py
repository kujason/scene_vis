import numpy as np
import vtk

from core.builders.dataset_builder import DatasetBuilder
from datasets.kitti.obj import obj_utils, calib_utils

from scene_vis.vtk_wrapper import vtk_utils
from scene_vis.vtk_wrapper.vtk_pyramid_boxes import VtkPyramidBoxes
from scene_vis.vtk_wrapper.vtk_point_cloud import VtkPointCloud


def main():
    """Demo to display point clouds and 3D bounding boxes

    Keys:
        F1: Toggle LIDAR point cloud
        F2: Toggle depth map point cloud
        F3: Toggle labels
    """

    ##############################
    # Options
    ##############################
    # sample_name = '000050'
    # sample_name = '000095'
    # sample_name = '000641'
    # sample_name = '000169'
    # sample_name = '000191'
    # sample_name = '000197'
    # sample_name = '006562'
    # sample_name = '004965'
    sample_name = '001692'
    # sample_name = '000999'

    # Area extents
    x_min = -40
    x_max = 40
    y_min = -5
    y_max = 5
    z_min = 0
    z_max = 70

    ##############################
    # End of Options
    ##############################

    area_extents = np.array([[x_min, x_max], [y_min, y_max], [z_min, z_max]], dtype=np.float32)

    dataset = DatasetBuilder.build_kitti_obj_dataset(DatasetBuilder.KITTI_TRAINVAL)

    # Get sample info
    image = obj_utils.get_image(sample_name, dataset.image_2_dir)
    image_shape = image.shape[0:2]
    frame_calib = calib_utils.get_frame_calib(dataset.calib_dir, sample_name)

    # Get lidar points
    lidar_point_cloud, lidar_pc_i = obj_utils.get_lidar_point_cloud(
        sample_name, frame_calib, dataset.velo_dir, intensity=True)
    lidar_point_cloud, area_filter = obj_utils.filter_pc_to_area(lidar_point_cloud, area_extents)

    # Filter to image
    lidar_points_in_img = calib_utils.project_pc_to_image(
        lidar_point_cloud, frame_calib.p2)
    lidar_point_cloud, image_filter = obj_utils.filter_pc_to_image(
        lidar_point_cloud, lidar_points_in_img, image_shape)
    lidar_points_in_img_int = np.floor(lidar_points_in_img[:, image_filter]).astype(np.int32)
    lidar_point_colours = image[lidar_points_in_img_int[1], lidar_points_in_img_int[0]]

    # lidar_pc_i_valid = lidar_pc_i[area_filter]
    # lidar_point_colours = (np.repeat(lidar_pc_i_valid, 3).reshape(-1, 3) * 200 + 25).astype(np.uint8)

    # Get points from depth map
    depth_point_cloud = obj_utils.get_depth_map_point_cloud(
        sample_name, frame_calib, dataset.depth_dir)
    depth_point_cloud, area_filter = obj_utils.filter_pc_to_area(depth_point_cloud, area_extents)

    # Filter depth map points to area
    area_filter = np.reshape(area_filter, image.shape[0:2])
    depth_point_colours = image[area_filter]

    # Get bounding boxes
    gt_objects = obj_utils.read_labels(dataset.label_dir, sample_name)

    ##############################
    # Vtk Visualization
    ##############################
    vtk_pc_lidar = VtkPointCloud()
    vtk_pc_depth = VtkPointCloud()

    vtk_pc_lidar.vtk_actor.GetProperty().SetPointSize(2)
    vtk_pc_depth.vtk_actor.GetProperty().SetPointSize(2)

    vtk_pc_lidar.set_points(lidar_point_cloud.T, lidar_point_colours)
    vtk_pc_depth.set_points(depth_point_cloud.T, depth_point_colours)

    # Create VtkBoxes for boxes
    vtk_pyr_boxes = VtkPyramidBoxes()
    vtk_pyr_boxes.set_objects(gt_objects, vtk_utils.COLOUR_SCHEME_KITTI)

    # Create Axes
    axes = vtk.vtkAxesActor()
    axes.SetTotalLength(5, 5, 5)

    # Create Voxel Grid Renderer in bottom half
    vtk_renderer = vtk.vtkRenderer()
    # vtk_renderer.SetBackground(0.2, 0.3, 0.4)
    vtk_renderer.SetBackground(0.35, 0.45, 0.55)
    # vtk_renderer.SetBackground(0.1, 0.1, 0.1)
    # vtk_renderer.SetBackground(0.8, 0.8, 0.8)

    vtk_renderer.AddActor(vtk_pc_lidar.vtk_actor)
    vtk_renderer.AddActor(vtk_pc_depth.vtk_actor)

    vtk_renderer.AddActor(vtk_pyr_boxes.vtk_actor)
    # vtk_renderer.AddActor(axes)

    # Setup Camera
    current_cam = vtk_renderer.GetActiveCamera()
    current_cam.Pitch(170.0)
    current_cam.Roll(180.0)

    # Zooms out to fit all points on screen
    vtk_renderer.ResetCamera()

    # Zoom in slightly
    # current_cam.Zoom(2.5)

    # current_cam.SetViewAngle(5.0)

    # Reset the clipping range to show all points
    vtk_renderer.ResetCameraClippingRange()

    # Setup Render Window
    vtk_render_window = vtk.vtkRenderWindow()
    vtk_render_window.SetWindowName(
        "Point Cloud, Sample {}".format(sample_name))
    # vtk_render_window.SetSize(1280, 720)
    vtk_render_window.SetSize(960, 640)
    vtk_render_window.AddRenderer(vtk_renderer)

    # Setup custom interactor style, which handles mouse and key events
    vtk_render_window_interactor = vtk.vtkRenderWindowInteractor()
    vtk_render_window_interactor.SetRenderWindow(vtk_render_window)

    # Add custom interactor to toggle actor visibilities
    custom_interactor = vtk_utils.ToggleActorsInteractorStyle(
        [
            vtk_pc_lidar.vtk_actor,
            vtk_pc_depth.vtk_actor,
            vtk_pyr_boxes.vtk_actor
        ],
        vtk_renderer, current_cam, axes
    )

    vtk_render_window_interactor.SetInteractorStyle(custom_interactor)

    # Render in VTK
    vtk_render_window.Render()
    vtk_render_window_interactor.Start()  # Blocking
    # renderWindowInteractor.Initialize()   # Non-Blocking


if __name__ == "__main__":
    main()

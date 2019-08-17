import copy

import numpy as np
import vtk

import core
from core import box_3d_encoder, evaluation
from core import config_utils
from core.visualization.vtk_wrapper.vtk_boxes import VtkBoxes
from datasets.kitti.obj import obj_utils
from datasets.kitti.obj.obj_utils import Difficulty

COLOUR_SCHEME_PREDICTIONS = {
    "Easy GT": (255, 255, 0),     # Yellow
    "Medium GT": (255, 128, 0),   # Orange
    "Hard GT": (255, 0, 0),       # Red

    "Prediction": (50, 255, 50),  # Green
}


def get_point_cloud(pc_source, sample_name, frame_calib,
                    velo_dir=None, depth_dir=None, disp_dir=None,
                    image_shape=None, cam_idx=2):

    if pc_source == 'lidar':
        point_cloud = obj_utils.get_lidar_point_cloud_for_cam(
            sample_name, frame_calib, velo_dir, image_shape, cam_idx)
    elif pc_source == 'depth':
        point_cloud = obj_utils.get_depth_map_point_cloud(
            sample_name, frame_calib, depth_dir)
    elif pc_source == 'stereo':
        raise NotImplementedError('Not implemented yet')
    else:
        raise ValueError('Invalid point cloud source', pc_source)

    return point_cloud


def get_gts_based_on_difficulty(dataset, sample_name):
    """Returns lists of ground-truth based on difficulty.
    """
    # Get all ground truth labels and filter to dataset classes
    all_gt_objs = obj_utils.read_labels(dataset.label_dir, sample_name)
    gt_objs, _ = obj_utils.filter_labels_by_class(all_gt_objs, dataset.classes)

    # Filter objects to desired difficulty
    easy_gt_objs, _ = obj_utils.filter_labels_by_difficulty(
        copy.deepcopy(gt_objs), difficulty=Difficulty.EASY)
    medium_gt_objs, _ = obj_utils.filter_labels_by_difficulty(
        copy.deepcopy(gt_objs), difficulty=Difficulty.MODERATE)
    hard_gt_objs, _ = obj_utils.filter_labels_by_difficulty(
        copy.deepcopy(gt_objs), difficulty=Difficulty.HARD)

    for gt_obj in easy_gt_objs:
        gt_obj.type = 'Easy GT'
    for gt_obj in medium_gt_objs:
        gt_obj.type = 'Medium GT'
    for gt_obj in hard_gt_objs:
        gt_obj.type = 'Hard GT'

    return easy_gt_objs, medium_gt_objs, hard_gt_objs, all_gt_objs


def create_gt_vtk_boxes(easy_gt_objs, medium_gt_objs, hard_gt_objs,
                        all_gt_objs, show_orientations):
    vtk_hard_gt_boxes = VtkBoxes()
    vtk_medium_gt_boxes = VtkBoxes()
    vtk_easy_gt_boxes = VtkBoxes()
    vtk_all_gt_boxes = VtkBoxes()

    vtk_hard_gt_boxes.set_objects(
        hard_gt_objs, COLOUR_SCHEME_PREDICTIONS, show_orientations)
    vtk_medium_gt_boxes.set_objects(
        medium_gt_objs, COLOUR_SCHEME_PREDICTIONS, show_orientations)
    vtk_easy_gt_boxes.set_objects(
        easy_gt_objs, COLOUR_SCHEME_PREDICTIONS, show_orientations)
    vtk_all_gt_boxes.set_objects(
        all_gt_objs, VtkBoxes.COLOUR_SCHEME_KITTI, show_orientations)

    return vtk_easy_gt_boxes, vtk_medium_gt_boxes, vtk_hard_gt_boxes, vtk_all_gt_boxes


def get_max_ious_3d(all_gt_boxes_3d, pred_boxes_3d):
    """Helper function to calculate 3D IoU for the given predictions.

    Args:
        all_gt_boxes_3d: A list of the same ground-truth boxes in box_3d
            format.
        pred_boxes_3d: A list of predictions in box_3d format.
    """

    # Only calculate ious if there are predictions
    if pred_boxes_3d:
        # Convert to iou format
        gt_objs_iou_fmt = box_3d_encoder.box_3d_to_3d_iou_format(
            all_gt_boxes_3d)
        pred_objs_iou_fmt = box_3d_encoder.box_3d_to_3d_iou_format(
            pred_boxes_3d)

        max_ious_3d = np.zeros(len(all_gt_boxes_3d))
        for gt_obj_idx in range(len(all_gt_boxes_3d)):

            gt_obj_iou_fmt = gt_objs_iou_fmt[gt_obj_idx]

            ious_3d = evaluation.three_d_iou(gt_obj_iou_fmt,
                                             pred_objs_iou_fmt)

            max_ious_3d[gt_obj_idx] = np.amax(ious_3d)
    else:
        # No detections, all ious = 0
        max_ious_3d = np.zeros(len(all_gt_boxes_3d))

    return max_ious_3d


def setup_vtk_renderer():
    # Setup renderer
    vtk_renderer = vtk.vtkRenderer()
    vtk_renderer.SetBackground(0.2, 0.3, 0.4)

    return vtk_renderer


def setup_vtk_render_window(window_name, window_size, vtk_renderer):
    # Setup Render Window
    vtk_render_window = vtk.vtkRenderWindow()
    vtk_render_window.SetWindowName(window_name)
    vtk_render_window.SetSize(*window_size)
    vtk_render_window.AddRenderer(vtk_renderer)
    return vtk_render_window


def setup_vtk_interactor(vtk_render_window):
    # Setup custom interactor style, which handles mouse and key events
    vtk_render_window_interactor = vtk.vtkRenderWindowInteractor()
    vtk_render_window_interactor.SetRenderWindow(vtk_render_window)
    vtk_render_window_interactor.SetInteractorStyle(
        vtk.vtkInteractorStyleTrackballCamera())


def setup_vtk_camera(vtk_renderer):

    # Setup Camera
    current_cam = vtk_renderer.GetActiveCamera()
    current_cam.Pitch(170.0)
    current_cam.Roll(180.0)

    # Zooms out to fit all points on screen
    vtk_renderer.ResetCamera()

    # Zoom in slightly
    current_cam.Zoom(3.0)

    # Reset the clipping range to show all points
    vtk_renderer.ResetCameraClippingRange()

    return current_cam


def get_experiment_info(checkpoint_name):
    exp_output_base_dir = core.data_dir() + '/outputs/' + checkpoint_name

    # Parse experiment config
    config_file = exp_output_base_dir + '/{}.yaml'.format(checkpoint_name)
    config = config_utils.parse_yaml_config(config_file)

    predictions_base_dir = exp_output_base_dir + '/predictions'

    return config, predictions_base_dir

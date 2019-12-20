"""https://github.com/kujason/scene_vis"""

import os

import vtk

from core import demo_utils
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from scene_vis.vtk_wrapper import vtk_utils
from scene_vis.vtk_wrapper.vtk_point_cloud_glyph import VtkPointCloudGlyph


def main():
    nusc = NuScenes(version='v1.0-mini', dataroot='../../data/datasets/nuscenes', verbose=True)

    my_sample = nusc.sample[10]
    my_sample_token = my_sample['token']
    sample_record = nusc.get('sample', my_sample_token)

    pointsensor_token = sample_record['data']['LIDAR_TOP']
    camera_token = sample_record['data']['CAM_FRONT']

    # Get point cloud
    pointsensor = nusc.get('sample_data', pointsensor_token)
    pcl_path = os.path.join(nusc.dataroot, pointsensor['filename'])
    if pointsensor['sensor_modality'] == 'lidar':
        pc_data = LidarPointCloud.from_file(pcl_path)
    else:
        raise NotImplementedError()

    pc = pc_data.points[0:3]

    vtk_window_size = (1280, 720)
    vtk_renderer = demo_utils.setup_vtk_renderer()
    vtk_render_window = demo_utils.setup_vtk_render_window(
        'Point Cloud', vtk_window_size, vtk_renderer)

    vtk_interactor = vtk.vtkRenderWindowInteractor()
    vtk_interactor.SetRenderWindow(vtk_render_window)
    vtk_interactor.SetInteractorStyle(vtk_utils.ToggleActorsInteractorStyle(None, vtk_renderer))
    vtk_interactor.Initialize()

    vtk_pc = VtkPointCloudGlyph()
    vtk_pc.set_points(pc.T)
    vtk_renderer.AddActor(vtk_pc.vtk_actor)

    vtk_render_window.Render()
    vtk_interactor.Start()  # Blocking


if __name__ == '__main__':
    main()

"""https://github.com/kujason/scene_vis"""

import vtk


from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader

import core
from core import demo_utils
from scene_vis.vtk_wrapper.vtk_point_cloud_glyph import VtkPointCloudGlyph


def main():

    # set root_dir to the correct path to your dataset folder
    root_dir = core.data_dir() + '/datasets/argoverse/argoverse-tracking/sample/'

    vtk_window_size = (1280, 720)

    argoverse_loader = ArgoverseTrackingLoader(root_dir)

    print('Total number of logs:', len(argoverse_loader))
    argoverse_loader.print_all()

    for i, argoverse_data in enumerate(argoverse_loader):
        if i >= 3:
            break
        print(argoverse_data)

    argoverse_data = argoverse_loader[0]
    print(argoverse_data)

    argoverse_data = argoverse_loader.get('c6911883-1843-3727-8eaa-41dc8cda8993')
    print(argoverse_data)

    log_id = 'c6911883-1843-3727-8eaa-41dc8cda8993'  # argoverse_loader.log_list[55]
    frame_idx = 150
    camera = argoverse_loader.CAMERA_LIST[0]

    argoverse_data = argoverse_loader.get(log_id)

    city_name = argoverse_data.city_name

    print('-------------------------------------------------------')
    print(f'Log: {log_id}, \n\tframe: {frame_idx}, camera: {camera}')
    print('-------------------------------------------------------\n')

    lidar_points = argoverse_data.get_lidar(frame_idx)
    img = argoverse_data.get_image_sync(frame_idx, camera=camera)
    objects = argoverse_data.get_label_object(frame_idx)
    calib = argoverse_data.get_calibration(camera)

    # TODO: Calculate point colours

    vtk_pc = VtkPointCloudGlyph()
    vtk_pc.set_points(lidar_points)
    vtk_pc.set_point_size(2)

    vtk_renderer = demo_utils.setup_vtk_renderer()
    vtk_render_window = demo_utils.setup_vtk_render_window(
        'Argoverse Demo', vtk_window_size, vtk_renderer)

    vtk_axes = vtk.vtkAxesActor()
    vtk_axes.SetTotalLength(2, 2, 2)
    vtk_renderer.AddActor(vtk_axes)
    vtk_renderer.AddActor(vtk_pc.vtk_actor)

    current_cam = vtk_renderer.GetActiveCamera()
    current_cam.SetViewUp(0, 0, 1)
    current_cam.SetPosition(-50, 0, 15)
    current_cam.SetFocalPoint(30, 0, 0)

    vtk_interactor = demo_utils.setup_vtk_interactor(vtk_render_window)
    vtk_interactor.Start()


if __name__ == '__main__':
    main()

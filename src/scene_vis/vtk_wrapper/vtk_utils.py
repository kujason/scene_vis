"""https://github.com/kujason/scene_vis"""

import datetime
import vtk


COLOUR_SCHEME_KITTI = {
    "Car": (255, 0, 0),  # Red
    "Pedestrian": (255, 150, 50),  # Orange
    "Cyclist": (150, 50, 100),  # Purple

    "Van": (255, 150, 150),  # Peach
    "Person_sitting": (150, 200, 255),  # Sky Blue

    "Truck": (200, 200, 200),  # Light Grey
    "Tram": (150, 150, 150),  # Grey
    "Misc": (100, 100, 100),  # Dark Grey
    "DontCare": (255, 255, 255),  # White
}


class ToggleActorsInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
    """VTK interactor style that allows toggling the visibility of up to 12
    actors with the F1-F12 keys. This object should be initialized with
    the actors to toggle, and each actor will be assigned an F key based on
    its order in the list.
    """

    def __init__(self, actors, vtk_renderer, current_cam=None, axes=None,
                 vtk_win_to_img_filter=None, vtk_png_writer=None):
        super(ToggleActorsInteractorStyle).__init__()

        self.actors = actors
        self.AddObserver("KeyPressEvent", self.key_press_event)
        self.vtk_renderer = vtk_renderer
        self.current_cam = current_cam
        self.axes = axes

        # Screenshots
        self.vtk_win_to_img_filter = vtk_win_to_img_filter
        self.vtk_png_writer = vtk_png_writer

    def key_press_event(self, obj, event):
        vtk_render_window_interactor = self.GetInteractor()

        key = vtk_render_window_interactor.GetKeySym()
        if key in ["F1", "F2", "F3", "F4", "F5", "F6",
                   "F7", "F8", "F9", "F10", "F11", "F12"]:

            actor_idx = int(key.split("F")[1]) - 1

            if self.actors and actor_idx < len(self.actors) and \
                    self.actors[actor_idx] is not None:
                current_visibility = self.actors[actor_idx].GetVisibility()
                self.actors[actor_idx].SetVisibility(not current_visibility)
                self.vtk_renderer.GetRenderWindow().Render()

        elif key == 't':
            if self.vtk_renderer is not None and self.current_cam is not None:
                # Move camera to 0, 0, 0
                self.vtk_renderer.ResetCamera()
                self.current_cam.SetViewUp(0, -1, 0)
                self.current_cam.SetFocalPoint(0.0, 0.0, 20.0)
                self.current_cam.SetPosition(0.0, 0.0, 0.0)
                # self.current_cam.Zoom(0.8)
                self.current_cam.Zoom(0.55)
                self.vtk_renderer.ResetCameraClippingRange()
                self.vtk_renderer.GetRenderWindow().Render()

        elif key == 'a':
            if self.axes is not None:
                current_visibility = self.axes.GetVisibility()
                self.axes.SetVisibility(not current_visibility)
                self.vtk_renderer.GetRenderWindow().Render()

        elif key == 'c':
            camera = self.vtk_renderer.GetActiveCamera()
            print('{:.3f}, {:.3f}, {:.3f}'.format(*camera.GetPosition()))
            print('{:.3f}, {:.3f}, {:.3f}'.format(*camera.GetFocalPoint()))

        elif key == 's':
            if self.vtk_win_to_img_filter is not None:
                now = datetime.datetime.now()
                screenshot_name = 'screenshot_{}.png'.format(now)
                save_screenshot(screenshot_name,
                                self.vtk_win_to_img_filter, self.vtk_png_writer)
                print('Saved', screenshot_name)


def set_axes_font_size(axes, font_size):

    axes.GetXAxisCaptionActor2D().GetTextActor().SetTextScaleModeToNone()
    axes.GetXAxisCaptionActor2D().GetCaptionTextProperty().SetFontSize(font_size)
    axes.GetYAxisCaptionActor2D().GetTextActor().SetTextScaleModeToNone()
    axes.GetYAxisCaptionActor2D().GetCaptionTextProperty().SetFontSize(font_size)
    axes.GetZAxisCaptionActor2D().GetTextActor().SetTextScaleModeToNone()
    axes.GetZAxisCaptionActor2D().GetCaptionTextProperty().SetFontSize(font_size)


def setup_screenshots(vtk_render_window):
    vtk_win_to_img_filter = vtk.vtkWindowToImageFilter()
    vtk_win_to_img_filter.SetInput(vtk_render_window)

    vtk_png_writer = vtk.vtkPNGWriter()

    return vtk_win_to_img_filter, vtk_png_writer


def save_screenshot(file_path, vtk_win_to_img_filter, vtk_png_writer):
    """Saves a screenshot of the current render window

    Args:
        file_path: File path
        vtk_win_to_img_filter: Instance of vtkWindowToImageFilter
        vtk_png_writer: Instance of vtkPNGWriter
    """
    # Update filter
    vtk_win_to_img_filter.Modified()
    vtk_win_to_img_filter.Update()

    # Save png
    vtk_png_writer.SetFileName(file_path)
    vtk_png_writer.SetInputData(vtk_win_to_img_filter.GetOutput())
    vtk_png_writer.Write()

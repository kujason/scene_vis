import vtk


class ToggleActorsInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
    """VTK interactor style that allows toggling the visibility of up to 12
    actors with the F1-F12 keys. This object should be initialized with
    the actors to toggle, and each actor will be assigned an F key based on
    its order in the list.
    """

    def __init__(self, actors, vtk_renderer, current_cam=None, axes=None):
        super(ToggleActorsInteractorStyle).__init__()

        self.actors = actors
        self.AddObserver("KeyPressEvent", self.key_press_event)
        self.vtk_renderer = vtk_renderer

        self.current_cam = current_cam
        self.axes = axes

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
                self.current_cam.Zoom(0.6)
                self.vtk_renderer.ResetCameraClippingRange()
                self.vtk_renderer.GetRenderWindow().Render()

        elif key == 'a':
            if self.axes is not None:
                current_visibility = self.axes.GetVisibility()
                self.axes.SetVisibility(not current_visibility)
                self.vtk_renderer.GetRenderWindow().Render()


def set_axes_font_size(axes, font_size):

    axes.GetXAxisCaptionActor2D().GetTextActor().SetTextScaleModeToNone()
    axes.GetXAxisCaptionActor2D().GetCaptionTextProperty().SetFontSize(font_size)
    axes.GetYAxisCaptionActor2D().GetTextActor().SetTextScaleModeToNone()
    axes.GetYAxisCaptionActor2D().GetCaptionTextProperty().SetFontSize(font_size)
    axes.GetZAxisCaptionActor2D().GetTextActor().SetTextScaleModeToNone()
    axes.GetZAxisCaptionActor2D().GetCaptionTextProperty().SetFontSize(font_size)

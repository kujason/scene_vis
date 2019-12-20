"""https://github.com/kujason/scene_vis"""

import numpy as np
import vtk
from vtk.util import numpy_support


class VtkPointCloudGlyph:
    """(Experimental)
    """
    def __init__(self):

        # VTK Data
        self.vtk_poly_data = vtk.vtkPolyData()

        self.vtk_points = vtk.vtkPoints()

        # Colours for each point in the point cloud
        self.vtk_colours = None
        self.vtk_points_temp = vtk.vtkPoints()
        # self.something = None
        self.vtk_id_list = vtk.vtkIdList()

        # self.ranges = None
        # self.vtk_id_list = vtk.vtkIdList()

        # self.vtk_colours = vtk.vtkUnsignedCharArray()
        # self.vtk_colours.SetNumberOfComponents(3)
        # self.vtk_colours.SetName("Colours")

        self.points = np.zeros((0, 3))
        self.colours = []

        self.point_idx = 0

        # Poly Data
        self.vtk_poly_data.SetPoints(self.vtk_points)

        self.glyphFilter = vtk.vtkVertexGlyphFilter()
        self.glyphFilter.AddInputData(self.vtk_poly_data)
        self.glyphFilter.Update()

        # Poly Data Mapper
        self.vtk_poly_data_mapper = vtk.vtkPolyDataMapper()
        self.vtk_poly_data_mapper.SetInputConnection(self.glyphFilter.GetOutputPort())

        # Actor
        self.vtk_actor = vtk.vtkActor()
        self.vtk_actor.SetMapper(self.vtk_poly_data_mapper)

    def set_points(self, points, point_colours=None):
        """Sets the point cloud to be visualized

        Args:
            points: (N, 3) List of points
            point_colours: (N, 3) BGR pixel colours corresponding to each point from OpenCV
        """

        # Set the points
        flattened_points = np.asarray(points, np.float32).flatten()
        np_to_vtk_points = numpy_support.numpy_to_vtk(
            flattened_points, deep=True, array_type=vtk.VTK_TYPE_FLOAT32)
        np_to_vtk_points.SetNumberOfComponents(3)

        self.vtk_points.SetData(np_to_vtk_points)

        if point_colours is not None:
            # Set point colours if provided
            # Rearrange OpenCV BGR into RGB format
            point_colours = np.asarray(point_colours, np.uint8)[:, [2, 1, 0]]

            # Set the point colours
            flattened_colours = point_colours.flatten()
            self.vtk_colours = numpy_support.numpy_to_vtk(
                flattened_colours, deep=True, array_type=vtk.VTK_TYPE_UINT8)
            self.vtk_colours.SetNumberOfComponents(3)

        else:
            # Use heights if no colours provided
            y_min = np.amin(points, axis=0)[1]
            y_max = np.amax(points, axis=0)[1]
            y_range = y_max - y_min

            if y_range > 0:
                pts_y = (points.transpose()[1] - y_min) / y_range
            else:
                pts_y = y_min

            height_array = pts_y.astype(np.float32)
            self.vtk_colours = numpy_support.numpy_to_vtk(
                height_array, deep=True, array_type=vtk.VTK_TYPE_FLOAT32)
            self.vtk_colours.SetNumberOfComponents(1)

            # Update PolyDataMapper to display height scalars
            self.vtk_poly_data_mapper.SetColorModeToDefault()
            self.vtk_poly_data_mapper.SetScalarRange(0, 1.0)
            self.vtk_poly_data_mapper.SetScalarVisibility(1)

        # Set point colours in Poly Data
        self.vtk_poly_data.GetPointData().SetScalars(self.vtk_colours)

    def set_point_size(self, point_size):
        self.vtk_actor.GetProperty().SetPointSize(point_size)

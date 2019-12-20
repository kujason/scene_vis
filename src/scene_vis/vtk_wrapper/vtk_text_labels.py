"""https://github.com/kujason/scene_vis"""

import vtk


class VtkTextLabels:
    """Display text labels at 3D positions
    """

    def __init__(self):

        self.vtk_actor = vtk.vtkActor2D()

    def set_text_labels(self, positions_3d, text_labels, colour=None):
        """Set text labels for each 3D position

        Args:
            positions_3d: (N, 3) List of 3D positions
            text_labels: List of text strings for each 3D position
            colour: (optional) text colour tuple (r, g, b)
        """

        # Create vtk data
        vtk_poly_data = vtk.vtkPolyData()
        vtk_points = vtk.vtkPoints()
        vtk_cell_array = vtk.vtkCellArray()

        label_array = vtk.vtkStringArray()
        label_array.SetName('label')

        for point_idx in range(len(positions_3d)):
            point = tuple(positions_3d[point_idx])

            vtk_points.InsertNextPoint(*point)
            vtk_cell_array.InsertNextCell(1)
            vtk_cell_array.InsertCellPoint(point_idx)
            label_array.InsertNextValue(text_labels[point_idx])

        vtk_poly_data.SetPoints(vtk_points)
        vtk_poly_data.SetVerts(vtk_cell_array)

        vtk_poly_data.GetPointData().AddArray(label_array)

        hierarchy = vtk.vtkPointSetToLabelHierarchy()
        hierarchy.SetInputData(vtk_poly_data)
        hierarchy.SetLabelArrayName('label')

        if colour is not None:
            hierarchy.GetTextProperty().SetColor(colour)
        else:
            # Default colour (almost white)
            hierarchy.GetTextProperty().SetColor((0.9, 0.9, 0.9))
            hierarchy.GetTextProperty().SetFontSize(12)

        placement_mapper = vtk.vtkLabelPlacementMapper()
        placement_mapper.SetInputConnection(hierarchy.GetOutputPort())
        placement_mapper.GeneratePerturbedLabelSpokesOn()

        # Add rounded rectangular background
        placement_mapper.SetShapeToRoundedRect()
        placement_mapper.SetBackgroundColor(0.2, 0.2, 0.2)
        placement_mapper.SetBackgroundOpacity(0.5)
        placement_mapper.SetMargin(5)

        self.vtk_actor.SetMapper(placement_mapper)

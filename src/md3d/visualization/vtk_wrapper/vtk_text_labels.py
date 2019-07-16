import vtk


class VtkTextLabels:
    """
    VtkTextLabels can be used to display text labels at 3D positions
    """

    def __init__(self):

        self.vtk_actor = vtk.vtkActor2D()

    def set_text_labels(self, positions_3d, text_labels, colour=None):
        """Set text labels for each 3D position

        :param positions_3d: list of 3D positions N x [x, y, z]
        :param text_labels: list of text strings for each 3D position
        :param colour: (optional) text colour tuple (r, g, b)
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

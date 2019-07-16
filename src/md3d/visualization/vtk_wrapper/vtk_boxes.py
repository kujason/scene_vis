import numpy as np
import vtk

# from interdepth.datasets.kitti_obj import box_utils
from md3d.core import box_3d_encoder


class VtkBoxes:
    """
    VtkBoxes displays coloured 3d boxes
    """

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

    def __init__(self):

        # VTK Data
        self.vtk_poly_data = vtk.vtkPolyData()
        self.vtk_points = vtk.vtkPoints()
        self.vtk_cells = vtk.vtkCellArray()

        self.vtk_poly_data.SetPoints(self.vtk_points)
        self.vtk_poly_data.SetVerts(self.vtk_cells)
        self.vtk_poly_data.Modified()

        # Cube Source
        self.vtk_cube_source = vtk.vtkCubeSource()

        # Glyph 3D
        self.vtk_glyph_3d = vtk.vtkGlyph3D()
        self.vtk_glyph_3d.SetSourceConnection(
            self.vtk_cube_source.GetOutputPort())
        self.vtk_glyph_3d.SetInputData(self.vtk_poly_data)
        self.vtk_glyph_3d.ScalingOff()
        self.vtk_glyph_3d.Update()

        # Data Set Mapper
        self.vtk_data_set_mapper = vtk.vtkDataSetMapper()

        # Actor for Boxes
        self.vtk_actor = vtk.vtkActor()
        self.vtk_actor.SetMapper(self.vtk_data_set_mapper)
        self.vtk_actor.GetProperty().SetRepresentationToWireframe()
        self.vtk_actor.GetProperty().SetLineWidth(2)

        # Actor for Orientation Lines
        self.vtk_lines_actor = vtk.vtkActor()
        self.vtk_lines_actor.GetProperty().SetLineWidth(2)

    def set_line_width(self, line_width):
        self.vtk_actor.GetProperty().SetLineWidth(line_width)
        self.vtk_lines_actor.GetProperty().SetLineWidth(line_width)

    def set_objects(self, object_labels, colour_scheme=None,
                    show_orientations=False):
        """Parses a list of boxes_3d and sets their colours

        Args:
            object_labels: list of ObjectLabels
            colour_scheme: colours for each class (e.g. COLOUR_SCHEME_KITTI)
            show_orientations: (optional) if True, self.vtk_lines_actor
                can be used to display the orientations of each box
        """
        box_corners = []
        box_colours = []

        orientation_vectors = None
        if show_orientations:
            orientation_vectors = []

        for obj in object_labels:

            # Ignore DontCare boxes since they are at (-1000, -1000, -1000)
            if obj.type == 'DontCare':
                continue

            box_3d = box_3d_encoder.object_label_to_box_3d(obj)

            # Get box corners
            corners = np.array(box_3d_encoder.compute_box_3d_corners(box_3d))
            if corners.size != 0:
                box_corners.append(corners.transpose())

                # Get colours
                if colour_scheme is not None:
                    if obj.type in colour_scheme:
                        box_colours.append(colour_scheme[obj.type])
                    else:
                        # Default (White)
                        box_colours.append((255, 255, 255))

                if show_orientations:
                    # Overwrite self.vtk_actor to contain both actors
                    vtk_boxes_actor = self.vtk_actor
                    self.vtk_actor = vtk.vtkAssembly()
                    self.vtk_actor.AddPart(vtk_boxes_actor)
                    self.vtk_actor.AddPart(self.vtk_lines_actor)

                    # Distance off the ground
                    arrow_height = obj.h / 2.0

                    # Start and end points
                    arrow_start = np.add(obj.t, [0, -arrow_height, 0])
                    arrow_end = np.add(obj.t, [obj.l * np.cos(obj.ry),
                                               -arrow_height,
                                               obj.l * -np.sin(obj.ry)])
                    orientation_vectors.append([arrow_start, arrow_end])

        self._set_boxes(box_corners, box_colours, orientation_vectors)

    def _set_boxes(self, box_corners_list, box_colours=None,
                   orientation_vectors=None):
        """
        Sets the box corners and box colours for display

        :param box_corners_list: list of box corners N x [8 x [x,y,z]]
        :param box_colours: (optional) list of unsigned char colour tuples
        :param orientation_vectors: (optional) vectors representing
            box orientations
        """

        vtk_boxes = vtk.vtkUnstructuredGrid()

        vtk_points = vtk.vtkPoints()

        # Char Array for cell colours
        vtk_cell_colours = vtk.vtkUnsignedCharArray()
        vtk_cell_colours.SetNumberOfComponents(3)

        current_point_id = 0

        # Orientations
        if orientation_vectors:
            vtk_line_points = vtk.vtkPoints()
            vtk_lines_cell_array = vtk.vtkCellArray()

            current_line_point_id = 0

        for box_idx in range(len(box_corners_list)):

            box_corners = box_corners_list[box_idx]

            # Sort corners into the order required by VTK
            box_x = box_corners[:, 0]
            box_y = box_corners[:, 1]
            box_z = box_corners[:, 2]

            sorted_order = np.lexsort((box_x, box_y, box_z))
            box_corners = box_corners[sorted_order]

            vtk_box = vtk.vtkVoxel()

            for i in range(len(box_corners)):
                point = box_corners[i]
                vtk_points.InsertNextPoint(*point)

                vtk_box.GetPointIds().SetId(i, current_point_id)
                current_point_id += 1

            vtk_boxes.InsertNextCell(vtk_box.GetCellType(),
                                     vtk_box.GetPointIds())

            # Set cell colour
            if box_colours:
                vtk_cell_colours.InsertNextTuple(box_colours[box_idx])

            # Setup orientation lines
            if orientation_vectors:
                arrow_start = orientation_vectors[box_idx][0]
                arrow_end = orientation_vectors[box_idx][1]

                vtk_line_points.InsertNextPoint(arrow_start)
                vtk_line_points.InsertNextPoint(arrow_end)

                vtk_line = vtk.vtkLine()
                vtk_line.GetPointIds().SetId(0, current_line_point_id)
                vtk_line.GetPointIds().SetId(1, current_line_point_id + 1)
                current_line_point_id += 2

                vtk_lines_cell_array.InsertNextCell(vtk_line)

        vtk_boxes.SetPoints(vtk_points)
        vtk_boxes.GetCellData().SetScalars(vtk_cell_colours)
        self.vtk_data_set_mapper.SetInputData(vtk_boxes)

        if orientation_vectors:
            vtk_lines_poly_data = vtk.vtkPolyData()
            vtk_lines_poly_data.SetPoints(vtk_line_points)
            vtk_lines_poly_data.SetLines(vtk_lines_cell_array)
            vtk_lines_poly_data.GetCellData().SetScalars(vtk_cell_colours)

            vtk_lines_mapper = vtk.vtkPolyDataMapper()
            vtk_lines_mapper.SetInputData(vtk_lines_poly_data)

            self.vtk_lines_actor.SetMapper(vtk_lines_mapper)


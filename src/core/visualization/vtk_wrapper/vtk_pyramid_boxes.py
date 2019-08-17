"""https://github.com/kujason/scene_vis"""

import numpy as np
import vtk

from core import box_3d_encoder


class VtkPyramidBoxes:
    """Displays coloured 3d boxes with front pyramid to show orientation
    """

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

        # Char Array for box colours
        self.vtk_box_colours = vtk.vtkUnsignedCharArray()
        self.vtk_box_colours.SetNumberOfComponents(3)

        # VTK Pyramid Lines Data
        self.vtk_pyramid_points = vtk.vtkPoints()
        self.vtk_pyramid_lines = vtk.vtkCellArray()

        self.vtk_pyramid_poly_data = vtk.vtkPolyData()
        self.vtk_pyramid_poly_data.SetPoints(self.vtk_points)
        self.vtk_pyramid_poly_data.SetVerts(self.vtk_cells)
        self.vtk_pyramid_poly_data.Modified()

        self.vtk_pyramid_actor = vtk.vtkActor()
        self.vtk_pyramid_actor.GetProperty().SetLineWidth(2)

        # Char Array for pyramid colours
        self.vtk_pyramid_colours = vtk.vtkUnsignedCharArray()
        self.vtk_pyramid_colours.SetNumberOfComponents(3)

    def set_line_width(self, line_width):
        self.vtk_actor.GetProperty().SetLineWidth(line_width)
        self.vtk_pyramid_actor.GetProperty().SetLineWidth(line_width)

    def set_objects(self, object_labels, colour_scheme=None, show_orientations=True):
        """Parses a list of ObjectLabel to set boxes and their colours

        Args:
            object_labels: List of ObjectLabels to visualize
            colour_scheme: colours for each class (e.g. vtk_utils.COLOUR_SCHEME_KITTI)
            show_orientations:

        Returns:

        """
        """

        :param object_labels: 
        :param colour_scheme: 
        :param show_orientations: (optional) if True, show box orientations
        """
        box_corners = []
        box_colours = []

        pyramid_tips = None
        if show_orientations:
            pyramid_tips = []

        for obj in object_labels:

            # Ignore DontCare boxes since they are at (-1000, -1000, -1000)
            if obj.type == 'DontCare':
                continue

            # Get box corners
            box_3d = box_3d_encoder.object_label_to_box_3d(obj)
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
                    self.vtk_actor.AddPart(self.vtk_pyramid_actor)

                    # Distance off the ground
                    arrow_height = obj.h / 2.0
                    pyramid_tip_length = obj.l * 0.2
                    half_obj_l = obj.l / 2.0
                    # Distance from centroid
                    pyramid_tip_dist = half_obj_l + pyramid_tip_length

                    # Start and end points
                    # arrow_start = np.add(obj.t, [0, -arrow_height, 0])
                    pyramid_tip =\
                        np.add(obj.t, [pyramid_tip_dist * np.cos(obj.ry),
                                       -arrow_height,
                                       pyramid_tip_dist * -np.sin(obj.ry)])
                    pyramid_tips.append(pyramid_tip)

        self._set_box_corners(box_corners, box_colours, pyramid_tips)

    def _set_box_corners(self, box_corners_list, box_colours=None, pyramid_tips=None):
        """Sets the box corners and box colours for display

        Args:
            box_corners_list: list of box corners N x [8 x [x,y,z]]
            box_colours: (optional) list of unsigned char colour tuples
            pyramid_tips: (optional) pyramid tip positions N x [x, y, z]
        """
        vtk_boxes = vtk.vtkUnstructuredGrid()
        vtk_box_points = vtk.vtkPoints()

        current_point_id = 0

        for box_idx in range(len(box_corners_list)):

            box_corners = box_corners_list[box_idx]

            # Create pyramid
            pyramid_tip = pyramid_tips[box_idx]
            self._create_pyramid_points(pyramid_tip, box_corners)
            self._create_pyramid_lines(box_idx)

            # Sort corners into the order required by VTK
            box_x = box_corners[:, 0]
            box_y = box_corners[:, 1]
            box_z = box_corners[:, 2]

            sorted_order = np.lexsort((box_x, box_y, box_z))
            box_corners = box_corners[sorted_order]

            vtk_box = vtk.vtkVoxel()

            for i in range(len(box_corners)):
                point = box_corners[i]
                vtk_box_points.InsertNextPoint(*point)

                vtk_box.GetPointIds().SetId(i, current_point_id)
                current_point_id += 1

            vtk_boxes.InsertNextCell(vtk_box.GetCellType(),
                                     vtk_box.GetPointIds())

            # Set cell colour
            if box_colours:
                self.vtk_box_colours.InsertNextTuple(box_colours[box_idx])

                # Set colours for 4 pyramid lines
                for i in range(4):
                    self.vtk_pyramid_colours.InsertNextTuple(
                        box_colours[box_idx])

        vtk_boxes.SetPoints(vtk_box_points)
        vtk_boxes.GetCellData().SetScalars(self.vtk_box_colours)
        self.vtk_data_set_mapper.SetInputData(vtk_boxes)

        # Setup pyramid poly data
        self.vtk_pyramid_poly_data.SetPoints(self.vtk_pyramid_points)
        self.vtk_pyramid_poly_data.SetLines(self.vtk_pyramid_lines)
        vtk_pyramid_lines_mapper = vtk.vtkPolyDataMapper()
        vtk_pyramid_lines_mapper.SetInputData(self.vtk_pyramid_poly_data)
        self.vtk_pyramid_actor.SetMapper(vtk_pyramid_lines_mapper)

        self.vtk_pyramid_poly_data.GetCellData().SetScalars(
            self.vtk_pyramid_colours)

    def _create_pyramid_points(self, pyramid_tip, box_corners):
        """Adds pyramid points to self.vtk_pyramid_points.
        p0: pyramid tip
        p1-p4: front corners of box

        Args:
            pyramid_tip: pyramid tip position (x, y, z)
            box_corners: point corners of the box
        """

        # Pyramid tip
        p0 = pyramid_tip

        # Front corners
        p1 = box_corners[0]
        p2 = box_corners[1]
        p3 = box_corners[4]
        p4 = box_corners[5]

        self.vtk_pyramid_points.InsertNextPoint(p0)
        self.vtk_pyramid_points.InsertNextPoint(p1)
        self.vtk_pyramid_points.InsertNextPoint(p2)
        self.vtk_pyramid_points.InsertNextPoint(p3)
        self.vtk_pyramid_points.InsertNextPoint(p4)

    def _create_pyramid_lines(self, current_box_idx):
        """Collects points into lines for the pyramid

        Args:
            current_box_idx: Current box index, used for point indexing
        """

        point_idx_start = current_box_idx * 5

        line1 = vtk.vtkLine()
        line1.GetPointIds().SetId(0, point_idx_start + 1)
        line1.GetPointIds().SetId(1, point_idx_start)

        line2 = vtk.vtkLine()
        line2.GetPointIds().SetId(0, point_idx_start + 2)
        line2.GetPointIds().SetId(1, point_idx_start)

        line3 = vtk.vtkLine()
        line3.GetPointIds().SetId(0, point_idx_start + 3)
        line3.GetPointIds().SetId(1, point_idx_start)

        line4 = vtk.vtkLine()
        line4.GetPointIds().SetId(0, point_idx_start + 4)
        line4.GetPointIds().SetId(1, point_idx_start)

        self.vtk_pyramid_lines.InsertNextCell(line1)
        self.vtk_pyramid_lines.InsertNextCell(line2)
        self.vtk_pyramid_lines.InsertNextCell(line3)
        self.vtk_pyramid_lines.InsertNextCell(line4)

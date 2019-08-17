"""https://github.com/kujason/scene_vis"""

import numpy as np
import vtk


class VtkFrustums:
    """Frustums
    """

    def __init__(self):

        # VTK Data
        self.vtk_poly_data = vtk.vtkPolyData()
        self.vtk_points = vtk.vtkPoints()
        self.vtk_cells = vtk.vtkCellArray()

        self.vtk_poly_data.SetPoints(self.vtk_points)
        self.vtk_poly_data.SetVerts(self.vtk_cells)
        self.vtk_poly_data.Modified()

        # Data Set Mapper
        self.vtk_data_set_mapper = vtk.vtkDataSetMapper()

        # Actor for Boxes
        self.vtk_actor = vtk.vtkActor()
        self.vtk_actor.SetMapper(self.vtk_data_set_mapper)

    def _make_quad(self, ids):
        polygon = vtk.vtkPolygon()
        polygon.GetPointIds().SetNumberOfIds(4)
        for i in range(4):
            polygon.GetPointIds().SetId(i, ids[i])

        return polygon

    def get_frustum_points(self, obj, stereo_calib,
                           depth_lo=5.0, depth_hi=70.0, x_offset=0.075):
        """ Calculates frustum points. Points are numbered starting

        :param obj: ObjectLabel
        :param stereo_calib: FrameCalibrationData stereo calibration
        :param depth_lo: frustum closer depth
        :param depth_hi: frustum farther depth
        :param x_offset: offset to correct for kitti stereo projection error

        :return: p0 - p7 frustum points
        """

        # Get 2D points from obj
        uv0 = (obj.x2, obj.y1)
        uv1 = (obj.x2, obj.y2)
        uv2 = (obj.x1, obj.y2)
        uv3 = (obj.x1, obj.y1)

        temp_lo = depth_lo / stereo_calib.f
        temp_hi = depth_hi / stereo_calib.f

        # Re-project 2D points into 3D space
        x0 = (uv0[0] - stereo_calib.center_u) * temp_lo - x_offset
        x1 = (uv1[0] - stereo_calib.center_u) * temp_lo - x_offset
        x2 = (uv2[0] - stereo_calib.center_u) * temp_lo - x_offset
        x3 = (uv3[0] - stereo_calib.center_u) * temp_lo - x_offset
        x4 = (uv0[0] - stereo_calib.center_u) * temp_hi - x_offset
        x5 = (uv1[0] - stereo_calib.center_u) * temp_hi - x_offset
        x6 = (uv2[0] - stereo_calib.center_u) * temp_hi - x_offset
        x7 = (uv3[0] - stereo_calib.center_u) * temp_hi - x_offset

        y0 = (uv0[1] - stereo_calib.center_v) * temp_lo
        y1 = (uv1[1] - stereo_calib.center_v) * temp_lo
        y2 = (uv2[1] - stereo_calib.center_v) * temp_lo
        y3 = (uv3[1] - stereo_calib.center_v) * temp_lo
        y4 = (uv0[1] - stereo_calib.center_v) * temp_hi
        y5 = (uv1[1] - stereo_calib.center_v) * temp_hi
        y6 = (uv2[1] - stereo_calib.center_v) * temp_hi
        y7 = (uv3[1] - stereo_calib.center_v) * temp_hi

        p0 = [x0, y0, depth_lo]
        p1 = [x1, y1, depth_lo]
        p2 = [x2, y2, depth_lo]
        p3 = [x3, y3, depth_lo]
        p4 = [x4, y4, depth_hi]
        p5 = [x5, y5, depth_hi]
        p6 = [x6, y6, depth_hi]
        p7 = [x7, y7, depth_hi]

        return p0, p1, p2, p3, p4, p5, p6, p7

    def make_polygons(self, points, vtk_points, vtk_points_offset):

        for point_idx in range(8):
            vtk_points.InsertNextPoint(*points[point_idx])

        # Faces (front, back)
        polygon_f = self._make_quad(np.add([0, 1, 2, 3], vtk_points_offset))
        polygon_b = self._make_quad(np.add([4, 5, 6, 7], vtk_points_offset))

        # Faces (right, down, left, up)
        polygon_r = self._make_quad(np.add([0, 4, 5, 1], vtk_points_offset))
        polygon_d = self._make_quad(np.add([1, 5, 6, 2], vtk_points_offset))
        polygon_l = self._make_quad(np.add([2, 6, 7, 3], vtk_points_offset))
        polygon_u = self._make_quad(np.add([3, 7, 4, 0], vtk_points_offset))

        return vtk_points, [polygon_f, polygon_b,
                            polygon_r, polygon_d, polygon_l, polygon_u]

    def set_objects(self, obj_labels, stereo_calib):
        """
        Parses a list of ObjectLabel to set frustums and their colours

        :param obj_labels: list of ObjectLabels to visualize
        :param stereo_calib: stereo calibration
        """

        vtk_points = vtk.vtkPoints()
        vtk_polygons_cell_array = vtk.vtkCellArray()

        for obj_idx in range(len(obj_labels)):
            obj = obj_labels[obj_idx]
            frustum_points = self.get_frustum_points(
                obj, stereo_calib)
            vtk_points, vtk_polygons = self.make_polygons(
                frustum_points, vtk_points, obj_idx * 8)

            for polygon in vtk_polygons:
                vtk_polygons_cell_array.InsertNextCell(polygon)

        # Create a PolyData
        polygons_poly_data = vtk.vtkPolyData()
        polygons_poly_data.SetPoints(vtk_points)
        polygons_poly_data.SetPolys(vtk_polygons_cell_array)

        # Create a mapper and actor
        poly_data_mapper = vtk.vtkPolyDataMapper()
        poly_data_mapper.SetInputData(polygons_poly_data)

        self.vtk_actor.SetMapper(poly_data_mapper)
        self.vtk_actor.GetProperty().SetOpacity(0.8)

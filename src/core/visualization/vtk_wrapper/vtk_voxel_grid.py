"""https://github.com/kujason/scene_vis"""

import numpy as np
import vtk
from vtk.util import numpy_support


class VtkVoxelGrid:
    """Display cubes from a vtkCubeSource to visualize voxels from a VoxelGrid object.
    Scalar arrays such as height or point density can also be added and visualized.
    """

    def __init__(self):

        # Default Options
        self.use_heights_as_scalars = True

        # References to the converted numpy arrays to avoid seg faults
        self.np_to_vtk_points = None
        self.np_to_vtk_cells = None
        self.scalar_dict = {}

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
        self.vtk_glyph_3d.SetSourceConnection(self.vtk_cube_source.GetOutputPort())
        self.vtk_glyph_3d.SetInputData(self.vtk_poly_data)
        self.vtk_glyph_3d.ScalingOff()
        self.vtk_glyph_3d.Update()

        # Mapper
        self.vtk_poly_data_mapper = vtk.vtkPolyDataMapper()
        self.vtk_poly_data_mapper.SetColorModeToDefault()
        self.vtk_poly_data_mapper.SetScalarRange(0, 1.0)
        self.vtk_poly_data_mapper.SetScalarVisibility(True)
        self.vtk_poly_data_mapper.SetInputConnection(self.vtk_glyph_3d.GetOutputPort())

        # Voxel Grid Actor
        self.vtk_actor = vtk.vtkActor()
        self.vtk_actor.SetMapper(self.vtk_poly_data_mapper)

    def set_voxels(self, voxel_grid):
        """Sets the voxel positions to visualize

        Args:
            voxel_grid: VoxelGrid
        """

        # Get voxels from VoxelGrid
        voxels = voxel_grid.voxel_indices
        num_voxels = len(voxels)

        # Shift voxels based on extents and voxel size
        voxel_positions = (voxels + voxel_grid.min_voxel_coord) * voxel_grid.voxel_size
        voxel_positions += voxel_grid.voxel_size / 2.0

        # Resize the cube source based on voxel size
        self.vtk_cube_source.SetXLength(voxel_grid.voxel_size)
        self.vtk_cube_source.SetYLength(voxel_grid.voxel_size)
        self.vtk_cube_source.SetZLength(voxel_grid.voxel_size)

        # Set the voxels
        flattened_points = np.array(voxel_positions).flatten()
        flattened_points = flattened_points.astype(np.float32)
        self.np_to_vtk_points = numpy_support.numpy_to_vtk(
            flattened_points, deep=True, array_type=vtk.VTK_TYPE_FLOAT32)
        self.np_to_vtk_points.SetNumberOfComponents(3)
        self.vtk_points.SetData(self.np_to_vtk_points)

        # Save the heights as a scalar array
        if self.use_heights_as_scalars:
            self.set_scalar_array("Height", voxels.transpose()[1])
            self.set_active_scalars("Height")

        # Create cells, one per voxel, cells in the form: [length, point index]
        cell_lengths = np.ones(num_voxels)
        cell_indices = np.arange(0, num_voxels)
        flattened_cells = np.array([cell_lengths, cell_indices]).transpose().flatten()
        flattened_cells = flattened_cells.astype(np.int32)

        # Convert list of cells to vtk format and set the cells
        self.np_to_vtk_cells = numpy_support.numpy_to_vtk(
            flattened_cells, deep=True, array_type=vtk.VTK_ID_TYPE)
        self.np_to_vtk_cells.SetNumberOfComponents(2)
        self.vtk_cells.SetCells(num_voxels, self.np_to_vtk_cells)

    def set_scalar_array(self, scalar_name, scalars, scalar_range=None):
        """Sets a scalar array in the scalar_dict, which can be used
        to modify the colouring of each voxel.
        Use set_active_scalars to choose the scalar array to be visualized.
        If a scalar range is not given, the scalar array will be set based on
        the minimum and maximum scalar values.


        Args:
            scalar_name: Name of scalar array, used as dictionary key
            scalars: 1D array of scalar values corresponding to each cell
            scalar_range: (optional) Custom scalar range
        """

        if scalar_range is not None:
            range_min = scalar_range[0]
            range_max = scalar_range[1]

            if range_min == range_max:
                raise ValueError("Scalar range maximum cannot equal minimum")
            else:
                # Remap to range
                map_range = range_max - range_min
                remapped_scalar_values = (scalars - range_min) / map_range
                remapped_scalar_values = \
                    remapped_scalar_values.astype(np.float32)
        else:
            # Calculate scalar range if not specified
            scalar_min = np.amin(scalars)
            scalar_max = np.amax(scalars)

            if scalar_min == scalar_max:
                remapped_scalar_values = np.full(scalars.shape,
                                                 scalar_min,
                                                 dtype=np.float32)
            else:
                map_range = scalar_max - scalar_min
                remapped_scalar_values = (scalars - scalar_min) / map_range
                remapped_scalar_values = \
                    remapped_scalar_values.astype(np.float32)

        # Convert numpy array to vtk format
        vtk_scalar_array = numpy_support.numpy_to_vtk(
            remapped_scalar_values, deep=True, array_type=vtk.VTK_TYPE_FLOAT32)
        vtk_scalar_array.SetNumberOfComponents(1)
        vtk_scalar_array.SetName(scalar_name)

        # Add scalar array to the PolyData
        self.vtk_poly_data.GetPointData().AddArray(vtk_scalar_array)

        # Save the scalar array into a dict entry
        self.scalar_dict[scalar_name] = vtk_scalar_array

    def set_active_scalars(self, scalar_name):
        """Sets the active scalar array for display
        """
        self.vtk_poly_data.GetPointData().SetActiveScalars(scalar_name)

import os

import numpy as np
import vtk


class VtkImage:
    """Image
    """

    def __init__(self):

        self.vtk_actor = vtk.vtkImageActor()

        # Need to keep reference to the image
        self.image = None
        self.vtk_image_data = None

    def _save_image_data(self, vtk_image_data):
        self.vtk_image_data = vtk_image_data
        self.vtk_actor.SetInputData(vtk_image_data)

    def set_image(self, image):
        """Setup image actor from image data

        Args:
            image: RGB image array
        """

        # Flip vertically and change BGR->RGB
        image = np.copy(image)[::-1, :, ::-1]

        # Save reference to image
        self.image = np.ascontiguousarray(image, dtype=np.uint8)

        # Setup vtkImageImport
        height, width = image.shape[0:2]
        vtk_image_import = vtk.vtkImageImport()
        vtk_image_import.SetDataSpacing(1, 1, 1)
        vtk_image_import.SetDataOrigin(0, 0, 0)
        vtk_image_import.SetWholeExtent(0, width - 1, 0, height - 1, 0, 0)
        vtk_image_import.SetDataExtentToWholeExtent()
        vtk_image_import.SetDataScalarTypeToUnsignedChar()
        vtk_image_import.SetNumberOfScalarComponents(3)
        vtk_image_import.SetImportVoidPointer(self.image)
        vtk_image_import.Update()

        # Get vtkImageData
        vtk_image_data = vtk_image_import.GetOutput()
        self._save_image_data(vtk_image_data)

    def set_image_path(self, image_path):
        """Setup image actor from image at given path

        Args:
            image_path: path to image
        """

        # Check extension
        extension = os.path.splitext(image_path)[1]

        if extension == '.png':
            # Setup vtk image data
            vtk_png_reader = vtk.vtkPNGReader()
            vtk_png_reader.SetFileName(image_path)
            vtk_png_reader.Update()
            vtk_image_data = vtk_png_reader.GetOutput()

        else:
            raise NotImplementedError('Only .png images are supported, file was', extension)

        self._save_image_data(vtk_image_data)

    @staticmethod
    def center_camera(vtk_renderer, vtk_image_data):
        """Sets camera to fill render window with the image

        Args:
            vtk_renderer: vtkRenderer
            vtk_image_data: vtkImageData to calculate extents for centering
        """

        origin = vtk_image_data.GetOrigin()
        spacing = vtk_image_data.GetSpacing()
        extent = vtk_image_data.GetExtent()

        camera = vtk_renderer.GetActiveCamera()
        camera.ParallelProjectionOn()

        xc = origin[0] + 0.5 * (extent[0] + extent[1]) * spacing[0]
        yc = origin[1] + 0.5 * (extent[2] + extent[3]) * spacing[1]

        # xd = (extent[1] - extent[0] + 1) * spacing[0]
        yd = (extent[3] - extent[2] + 1) * spacing[1]

        d = camera.GetDistance()
        camera.SetParallelScale(0.5 * yd)
        camera.SetFocalPoint(xc, yc, 0.0)
        camera.SetPosition(xc, yc, d)

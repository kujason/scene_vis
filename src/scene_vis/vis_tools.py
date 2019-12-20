import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from datasets.kitti.obj import obj_utils

# Window sizes
CV2_SIZE_2_COL = (930, 280)
CV2_SIZE_3_COL = (620, 187)
CV2_SIZE_4_COL = (465, 140)


def project_pc_to_image(point_cloud, cam_p):
    """Projects a 3D point cloud to 2D points

    Args:
        point_cloud: (3, N) point cloud
        cam_p: 3x4 camera projection matrix

    Returns:
        pts_2d: (2, N) projected coordinates [u, v] of the 3D points
    """

    pc_padded = np.append(point_cloud, np.ones((1, point_cloud.shape[1])), axis=0)
    pts_2d = np.dot(cam_p, pc_padded)

    pts_2d[0:2] = pts_2d[0:2] / pts_2d[2]
    return pts_2d[0:2]


def project_corners_3d_to_image(corners_3d, p):
    """Computes the 3D bounding box projected onto
    image space.

    Keyword arguments:
    obj -- object file to draw bounding box
    p -- transform matrix

    Returns:
        corners: (N, 8, 2) Corner points projected into image space
        face_idx: 3D bounding box faces for drawing
    """
    # index for 3d bounding box face
    # it is converted to 4x4 matrix
    face_idx = np.array([0, 1, 5, 4,  # front face
                         1, 2, 6, 5,  # left face
                         2, 3, 7, 6,  # back face
                         3, 0, 4, 7]).reshape((4, 4))  # right face
    return project_pc_to_image(corners_3d, p), face_idx


def compute_box_3d_corners(box_3d):
    """Computes the 3D bounding box corner positions from a 3D box

    Args:
        box_3d: 3D box (x, y, z, l, w, h, ry)

    Returns:
        corners_3d:
    """

    tx, ty, tz, l, w, h, ry = box_3d

    # compute rotational matrix
    rot = np.array([[+np.cos(ry), 0, +np.sin(ry)],
                    [0, 1, 0],
                    [-np.sin(ry), 0, +np.cos(ry)]])

    # 3D BB corners
    x_corners = np.array(
        [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2])
    y_corners = np.array([0, 0, 0, 0, -h, -h, -h, -h])
    z_corners = np.array(
        [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2])

    corners_3d = np.dot(rot, np.array([x_corners, y_corners, z_corners]))

    corners_3d[0, :] = corners_3d[0, :] + tx
    corners_3d[1, :] = corners_3d[1, :] + ty
    corners_3d[2, :] = corners_3d[2, :] + tz

    return corners_3d


def project_orientation_3d(box_3d, cam_p):
    """Projects orientation vector given object and camera matrix

    Args:
        box_3d: 3D box (x, y, z, l, w, h, ry)
        cam_p: 3x4 camera projection matrix

    Returns:
        Projection of orientation vector into image
    """

    tx, ty, tz, l, w, h, ry = box_3d

    # Rotation matrix
    rot = np.array([[+np.cos(ry), 0, +np.sin(ry)],
                    [0, 1, 0],
                    [-np.sin(ry), 0, +np.cos(ry)]])

    orientation_3d = np.array([0.0, l, 0.0, 0.0, 0.0, 0.0]).reshape(3, 2)
    orientation_3d = np.dot(rot, orientation_3d)

    orientation_3d[0, :] = orientation_3d[0, :] + tx
    orientation_3d[1, :] = orientation_3d[1, :] + ty
    orientation_3d[2, :] = orientation_3d[2, :] + tz

    # only draw for boxes that are in front of the camera
    for idx in np.arange(orientation_3d.shape[1]):
        if orientation_3d[2, idx] < 0.1:
            return None

    return project_pc_to_image(orientation_3d, cam_p)


def plots_from_image(img,
                     subplot_rows=1,
                     subplot_cols=1,
                     display=True,
                     fig_size=None):
    """Forms the plot figure and axis for the visualization

    Args:
        img: image to plot
        subplot_rows: number of rows of the subplot grid
        subplot_cols: number of columns of the subplot grid
        display: display the image in non-blocking fashion
        fig_size: (optional) size of the figure
    """

    def set_plot_limits(axes, image):
        # Set the plot limits to the size of the image, y is inverted
        axes.set_xlim(0, image.shape[1])
        axes.set_ylim(image.shape[0], 0)

    if fig_size is None:
        img_shape = np.shape(img)
        fig_height = img_shape[0] / 100 * subplot_cols
        fig_width = img_shape[1] / 100 * subplot_rows
        fig_size = (fig_width, fig_height)

    # Create the figure
    fig, axes = plt.subplots(subplot_rows, subplot_cols, figsize=fig_size, sharex=True)
    fig.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, hspace=0.0)

    # Plot image
    if subplot_rows == 1 and subplot_cols == 1:
        # Single axis
        axes.imshow(img)
        set_plot_limits(axes, img)
    else:
        # Multiple axes
        for idx in range(axes.size):
            axes[idx].imshow(img)
            set_plot_limits(axes[idx], img)

    if display:
        plt.show(block=False)

    return fig, axes


def plots_from_sample_name(image_dir,
                           sample_name,
                           subplot_rows=1,
                           subplot_cols=1,
                           display=True,
                           fig_size=(15, 9.15)):
    """Forms the plot figure and axis for the visualization

    Args:
        image_dir: directory of image files in the wavedata
        sample_name: sample name of the image file to present
        subplot_rows: number of rows of the subplot grid
        subplot_cols: number of columns of the subplot grid
        display: display the image in non-blocking fashion
        fig_size: (optional) size of the figure
    """
    sample_name = int(sample_name)

    # Grab image data
    img = np.array(Image.open("{}/{:06d}.png".format(image_dir, sample_name)), dtype=np.uint8)

    # Create plot
    fig, axes = plots_from_image(img, subplot_rows, subplot_cols, display, fig_size)

    return fig, axes


def cv2_imshow(window_name, image,
               size_wh=None, row_col=None, location_xy=None):
    """Helper function for specifying window size and location when
        displaying images with cv2

    Args:
        window_name (string): Window title
        image: image to display
        size_wh: resize window
            Recommended sizes for 1920x1080 screen:
                2 col: (930, 280)
                3 col: (620, 187)
                4 col: (465, 140)
        row_col: Row and column to show images like subplots
        location_xy: location of window
    """

    if size_wh is not None:
        cv2.namedWindow(window_name, cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow(window_name, *size_wh)
    else:
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    if row_col is not None:
        start_x_offset = 60
        start_y_offset = 25
        y_offset = 28

        subplot_row = row_col[0]
        subplot_col = row_col[1]
        location_xy = (start_x_offset + subplot_col * size_wh[0],
                       start_y_offset + subplot_row * size_wh[1] + subplot_row * y_offset)

    if location_xy is not None:
        cv2.moveWindow(window_name, *location_xy)

    cv2.imshow(window_name, image)


def get_point_colours(points, cam_p, image):
    points_in_im = project_pc_to_image(points.T, cam_p)
    points_in_im_rounded = np.round(points_in_im).astype(np.int32)

    point_colours = image[points_in_im_rounded[1], points_in_im_rounded[0]]

    return point_colours


def draw_box_2d(ax, box_2d, color='#90EE900', linewidth=2):
    """Draws 2D boxes given coordinates in box_2d format

    Args:
        ax: subplot handle
        box_2d: ndarray containing box coordinates in box_2d format (y1, x1, y2, x2)
        color: color of box
    """
    box_x1 = box_2d[1]
    box_y1 = box_2d[0]
    box_w = box_2d[3] - box_x1
    box_h = box_2d[2] - box_y1

    rect = patches.Rectangle((box_x1, box_y1),
                             box_w, box_h,
                             linewidth=linewidth,
                             edgecolor=color,
                             facecolor='none')
    ax.add_patch(rect)


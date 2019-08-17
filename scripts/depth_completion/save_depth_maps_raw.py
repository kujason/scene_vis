import os
import sys
import time

import numpy as np

from core import transform_utils, depth_map_utils
from datasets.kitti.obj import obj_utils
from datasets.kitti.raw import raw_utils
from ip_basic import ip_basic


def main():
    """Interpolates the lidar point cloud using IP-Basic
    and saves a dense depth map of the scene.
    https://github.com/kujason/ip_basic
    """

    ##############################
    # Options
    ##############################

    kitti_raw_dir = os.path.expanduser('~/Kitti/raw')
    drive_id = '2011_09_26_drive_0039_sync'

    raw_data = raw_utils.get_raw_data(drive_id, kitti_raw_dir)

    # Fill algorithm ('ip_basic_{...}')
    fill_type = 'multiscale'

    save_depth_maps = True

    out_depth_map_dir = 'outputs/raw/{}/depth_02_{}'.format(drive_id, fill_type)

    ##############################
    # End of Options
    ##############################
    os.makedirs(out_depth_map_dir, exist_ok=True)

    # Rolling average array of times for time estimation
    avg_time_arr_length = 5
    last_fill_times = np.repeat([1.0], avg_time_arr_length)
    last_total_times = np.repeat([1.0], avg_time_arr_length)

    cam_p = raw_data.calib.P_rect_20

    image_shape = raw_data.get_cam2(0).size[::-1]

    frames_to_use = raw_data.velo_files
    num_frames = len(frames_to_use)

    for frame_idx, velo_path in enumerate(frames_to_use):

        # Calculate average time with last n fill times
        avg_fill_time = np.mean(last_fill_times)
        avg_total_time = np.mean(last_total_times)

        # Print progress
        sys.stdout.write('\rProcessing {} / {}, Avg Fill Time: {:.5f}s, '
                         'Avg Time: {:.5f}s, Est Time: {:.3f}s'.format(
                             frame_idx, num_frames - 1,
                             avg_fill_time, avg_total_time,
                             avg_total_time * (num_frames - frame_idx)))
        sys.stdout.flush()

        # Start timing
        start_total_time = time.time()

        # Load point cloud
        velo_points = raw_data.get_velo(frame_idx)[:, 0:3]
        velo_pc_padded = transform_utils.pad_pc(velo_points.T)

        cam0_point_cloud = raw_data.calib.R_rect_00 @ raw_data.calib.T_cam0_velo @ velo_pc_padded

        cam0_point_cloud, _ = obj_utils.filter_pc_to_area(
            cam0_point_cloud[0:3], area_extents=np.asarray([[-40, 40], [-3, 5], [0, 80]]))

        # Fill depth map
        if fill_type == 'multiscale':
            # Project point cloud to depth map
            projected_depths = depth_map_utils.project_depths(cam0_point_cloud, cam_p, image_shape)

            start_fill_time = time.time()
            final_depth_map, _ = ip_basic.fill_in_multiscale(projected_depths)
            end_fill_time = time.time()
        else:
            raise ValueError('Invalid fill algorithm')

        # Save depth maps
        if save_depth_maps:
            out_depth_map_path = out_depth_map_dir + '/{:010d}.png'.format(frame_idx)
            depth_map_utils.save_depth_map(out_depth_map_path, final_depth_map)

        # Stop timing
        end_total_time = time.time()

        # Update fill times
        last_fill_times = np.roll(last_fill_times, -1)
        last_fill_times[-1] = end_fill_time - start_fill_time

        # Update total times
        last_total_times = np.roll(last_total_times, -1)
        last_total_times[-1] = end_total_time - start_total_time


if __name__ == "__main__":
    main()

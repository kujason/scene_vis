import os

import numpy as np


def main():
    """
    train_mapping.txt
    date, drive_id, frame_idx

    train_rand.txt
    img_idx | line in train_mapping.txt (1 indexed)
    """

    train_mapping_path = os.path.expanduser(
        '~/Kitti/object/devkit_object/mapping/train_mapping.txt')
    train_rand_path = os.path.expanduser(
        '~/Kitti/object/devkit_object/mapping/train_rand_sep_lines.txt')

    train_mapping = np.loadtxt(train_mapping_path, dtype=np.str).reshape(-1, 3)
    train_rand = np.loadtxt(train_rand_path, dtype=np.str)

    val_split_path = os.path.expanduser('~/Kitti/object/val.txt')
    val_split = np.loadtxt(val_split_path, dtype=np.int32)

    # frame_range = np.arange(1, 7481)

    # 2011_09_26_drive_0014_sync
    # frame_range = np.arange(458, 631)

    # 2011_09_26_drive_0015_sync
    # frame_range = np.arange(750, 860)  # In training

    # frame_range = np.arange(1186, 1235)  #

    # 2011_10_03_drive_0047_sync highway sequence
    # frame_range = np.arange(3299, 3594)  # full sequence
    frame_range = np.arange(3426, 3498)  # 440-512 no skips

    all_sample_indices = []
    for frame_idx in frame_range:
        sample_idx = np.where(train_rand == str(frame_idx))[0][0]

        # Check if all in val
        val_sample_idx = np.where(val_split == sample_idx)
        in_val = np.shape(val_sample_idx)[1] > 0

        if not in_val:
            raise ValueError('Bad')

        print(sample_idx)

        all_sample_indices.append(sample_idx)

    np.savetxt('samples.txt', all_sample_indices, fmt='%d')
    print('Done')


if __name__ == '__main__':
    main()

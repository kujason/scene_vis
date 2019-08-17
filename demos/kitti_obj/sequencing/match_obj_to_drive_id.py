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

    train_rand_int = train_rand.astype(np.int32) - 1
    sample_to_raw_mapping = train_mapping[train_rand_int]

    # Prefix with sample name
    sample_names = ['{:06d}'.format(sample_idx) for sample_idx in range(len(train_mapping))]
    lines = np.concatenate([np.expand_dims(sample_names, 1), sample_to_raw_mapping], axis=1)

    # Save
    np.savetxt('outputs/obj_to_raw_mapping.txt', lines, fmt='%s')


if __name__ == '__main__':
    main()

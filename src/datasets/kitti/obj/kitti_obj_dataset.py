import fnmatch
import os

import numpy as np

from datasets.kitti.obj import obj_utils


class Sample:
    def __init__(self, name, augs):
        self.name = name
        self.augs = augs

    def __repr__(self):
        return '({}, augs: {})'.format(self.name, self.augs)


class KittiObjDataset:

    def __init__(self, dataset_config, train_val_test):

        self.dataset_config = dataset_config
        self.train_val_test = train_val_test

        # Parse config
        self.name = self.dataset_config.name

        self.data_split = self.dataset_config.data_split
        self.dataset_dir = os.path.expanduser(self.dataset_config.dataset_dir)
        data_split_dir = self.dataset_config.data_split_dir

        self.num_boxes = self.dataset_config.num_boxes
        self.num_angle_bins = self.dataset_config.num_angle_bins

        self.cam_idx = 2

        self.classes = list(self.dataset_config.classes)
        self.num_classes = len(self.classes)

        # Parse object filtering config
        obj_filter_config = self.dataset_config.obj_filter_config
        obj_filter_config.classes = self.classes
        self.obj_filter = obj_utils.ObjectFilter(obj_filter_config)

        self.has_labels = self.dataset_config.has_labels
        # self.cluster_split = self.config.cluster_split

        self.classes_name = self._set_up_classes_name()

        # Check that paths and split are valid
        self._check_dataset_dir()
        all_dataset_files = os.listdir(self.dataset_dir)
        self._check_data_split_valid(all_dataset_files)
        self.data_split_dir = self._check_data_split_dir_valid(
            all_dataset_files, data_split_dir)

        self.depth_version = self.dataset_config.depth_version

        # Setup directories
        self._set_up_directories()

        # Whether to oversample objects to required number of boxes
        self.oversample = self.dataset_config.oversample

        # Augmentation
        self.aug_config = self.dataset_config.aug_config

        # Initialize the sample list
        loaded_sample_names = self.load_sample_names(self.data_split)
        all_samples = [Sample(sample_name, []) for sample_name in loaded_sample_names]

        self.sample_list = np.asarray(all_samples)
        self.num_samples = len(self.sample_list)

        # Batch pointers
        self._index_in_epoch = 0
        self.epochs_completed = 0

    def _check_dataset_dir(self):
        """Checks that dataset directory exists in the file system

        Raises:
            FileNotFoundError: if the dataset folder is missing
        """
        # Check that dataset path is valid
        if not os.path.exists(self.dataset_dir):
            raise FileNotFoundError('Dataset path does not exist: {}'
                                    .format(self.dataset_dir))

    def _check_data_split_valid(self, all_dataset_files):
        possible_splits = []
        for file_name in all_dataset_files:
            if fnmatch.fnmatch(file_name, '*.txt'):
                possible_splits.append(os.path.splitext(file_name)[0])
        # This directory contains a readme.txt file, remove it from the list
        if 'readme' in possible_splits:
            possible_splits.remove('readme')

        if self.data_split not in possible_splits:
            raise ValueError("Invalid data split: {}, possible_splits: {}"
                             .format(self.data_split, possible_splits))

    def _check_data_split_dir_valid(self, all_dataset_files, data_split_dir):
        # Check data_split_dir
        # Get possible data split dirs from folder names in dataset folder
        possible_split_dirs = []
        for folder_name in all_dataset_files:
            if os.path.isdir(self.dataset_dir + '/' + folder_name):
                possible_split_dirs.append(folder_name)

        if data_split_dir in possible_split_dirs:
            # Overwrite with full path
            data_split_dir = self.dataset_dir + '/' + data_split_dir
        else:
            raise ValueError(
                "Invalid data split dir: {}, possible dirs".format(
                    data_split_dir, possible_split_dirs))

        return data_split_dir

    def _set_up_directories(self):
        """Sets up data directories."""
        # Setup Directories
        self.rgb_image_dir = self.data_split_dir + '/image_' + str(self.cam_idx)
        self.image_2_dir = self.data_split_dir + '/image_2'
        self.image_3_dir = self.data_split_dir + '/image_3'

        self.calib_dir = self.data_split_dir + '/calib'
        self.disp_dir = self.data_split_dir + '/disparity'
        self.planes_dir = self.data_split_dir + '/planes'
        self.velo_dir = self.data_split_dir + '/velodyne'
        self.depth_dir = self.data_split_dir + '/depth_{}_{}'.format(
            self.cam_idx, self.depth_version)

        if self.has_labels:
            self.label_dir = self.data_split_dir + '/label_2'

    def _set_up_classes_name(self):
        # Unique identifier for multiple classes
        if self.num_classes > 1:
            if self.classes == ['Pedestrian', 'Cyclist']:
                classes_name = 'People'
            elif self.classes == ['Car', 'Pedestrian', 'Cyclist']:
                classes_name = 'All'
            else:
                raise NotImplementedError('Need new unique identifier for '
                                          'multiple classes')
        else:
            classes_name = self.classes[0]

        return classes_name

    def get_sample_names(self):
        return [sample.name for sample in self.sample_list]

    def get_image_2_path(self, sample_name):
        return self.image_2_dir + '/' + sample_name + '.png'

    def get_image_3_path(self, sample_name):
        return self.image_3_dir + '/' + sample_name + '.png'

    def get_depth_map_path(self, sample_name):
        return self.depth_dir + '/' + sample_name + '.png'

    def get_velodyne_path(self, sample_name):
        return self.velo_dir + '/' + sample_name + '.bin'

    # Data loading methods
    def load_sample_names(self, data_split):
        """Load the sample names listed in this dataset's set file
        (e.g. train.txt, validation.txt)

        Args:
            data_split: the sample list to load

        Returns:
            A list of sample names (file names) read from
            the .txt file corresponding to the data split
        """
        set_file = self.dataset_dir + '/' + data_split + '.txt'
        with open(set_file, 'r') as f:
            sample_names = f.read().splitlines()

        return np.asarray(sample_names)

    def get_sample_dict(self, indices):
        """"""
        pass

    def _shuffle_samples(self):
        perm = np.arange(self.num_samples)
        np.random.shuffle(perm)
        self.sample_list = self.sample_list[perm]

    def next_batch(self, batch_size, shuffle):
        """
        Retrieve the next `batch_size` samples from this data set.

        Args:
            batch_size: number of samples in the batch
            shuffle: whether to shuffle the indices after an epoch is completed

        Returns:
            list of dictionaries containing sample information
        """

        # Create empty set of samples
        samples_in_batch = []

        start = self._index_in_epoch
        # Shuffle only for the first epoch
        if self.epochs_completed == 0 and start == 0 and shuffle:
            self._shuffle_samples()

        # Go to the next epoch
        if start + batch_size >= self.num_samples:

            # Finished epoch
            self.epochs_completed += 1

            # Get the rest examples in this epoch
            rest_num_examples = self.num_samples - start

            # Append those samples to the current batch
            samples_in_batch.extend(
                self.get_sample_dict(np.arange(start, self.num_samples)))

            # Shuffle the data
            if shuffle:
                self._shuffle_samples()

            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch

            # Append the rest of the batch
            samples_in_batch.extend(self.get_sample_dict(np.arange(start, end)))

        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch

            # Append the samples in the range to the batch
            samples_in_batch.extend(self.get_sample_dict(np.arange(start, end)))

        return samples_in_batch

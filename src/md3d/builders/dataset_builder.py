from copy import deepcopy

import yaml

import md3d
from md3d.core import config_utils
from md3d.datasets.kitti.kitti_obj_dataset import KittiObjDataset


class DatasetBuilder:
    """Class with static methods to return preconfigured dataset objects
    """

    CONFIG_DEFAULTS_YAML = \
        """
        dataset_type: 'kitti_obj'
        
        batch_size: 1
        oversample: True
        
        num_boxes: 32
        num_angle_bins: 32
        
        classes: ['Car']
        
        # Object Filtering
        obj_filter_config:
            # Note: Object types filtered based on classes
            difficulty_str: 'hard'
            box_2d_height: !!null
            truncation: !!null
            occlusion: !!null
        
        # Augmentation
        aug_config:
            box_jitter_type: 'oversample'  # 'oversample', 'all', !!null

        name: 'kitti'
        dataset_dir: '~/Kitti/object'
        data_split: 'train'
        data_split_dir: 'training'
        has_labels: True

        depth_version: 'multiscale'
        """

    KITTI_TRAIN = 'kitti_obj_train'
    KITTI_VAL = 'kitti_obj_val'
    KITTI_VAL_HALF = 'kitti_obj_val_half'
    KITTI_VAL_MINI = 'kitti_obj_val_mini'
    KITTI_TRAINVAL = 'kitti_obj_trainval'
    KITTI_TEST = 'kitti_obj_test'

    @staticmethod
    def get_config_obj(dataset_type):

        config_obj = config_utils.config_dict_to_object(
            yaml.load(DatasetBuilder.CONFIG_DEFAULTS_YAML))

        if dataset_type == DatasetBuilder.KITTI_TRAIN:
            return config_obj
        elif dataset_type == DatasetBuilder.KITTI_VAL:
            config_obj.data_split = 'val'
        elif dataset_type == DatasetBuilder.KITTI_VAL_HALF:
            config_obj.data_split = 'val_half'
        elif dataset_type == DatasetBuilder.KITTI_VAL_MINI:
            config_obj.data_split = 'val_mini'
        elif dataset_type == DatasetBuilder.KITTI_TRAINVAL:
            config_obj.data_split = 'trainval'
        elif dataset_type == DatasetBuilder.KITTI_TEST:
            config_obj.data_split = 'test'
            config_obj.data_split_dir = 'testing'
            config_obj.has_labels = False
        else:
            raise ValueError('Invalid dataset type', dataset_type)

        return config_obj

    @staticmethod
    def build_kitti_obj_dataset(dataset_config, train_val_test='train'):

        if isinstance(dataset_config, str):
            config_obj = DatasetBuilder.get_config_obj(dataset_config)
            return KittiObjDataset(config_obj, train_val_test)

        return KittiObjDataset(dataset_config, train_val_test)


if __name__ == '__main__':

    train_dataset = DatasetBuilder.build_kitti_obj_dataset(DatasetBuilder.KITTI_TRAIN)
    print(train_dataset)

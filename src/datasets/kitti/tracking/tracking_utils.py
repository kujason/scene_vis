from collections import namedtuple
import os
import re

import numpy as np
import pykitti

from datasets.kitti.obj.obj_utils import ObjectLabel


def load_poses():
    pass


def load_calib(calib_path):
    """Read in a calibration file and parse into a dictionary."""
    data = {}

    with open(calib_path, 'r') as f:
        for line in f.readlines():
            # key, value = line.split(':', 1)
            key, value = re.compile('[: ]*').split(line, 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    # return namedtuple(data.keys(), data.values())
    return data


class Tracklet:

    def __init__(self):
        self.frame = None
        self.id = None
        self.cls = None
        self.trunc = 0.0
        self.occ = 0
        self.alpha = 0.0
        self.x1 = 0.0
        self.y1 = 0.0
        self.x2 = 0.0
        self.y2 = 0.0
        self.h = 0.0
        self.w = 0.0
        self.l = 0.0
        self.t = (0.0, 0.0, 0.0)
        self.ry = 0.0

    def __repr__(self):

        return '(f:{}, id:{}, c:{}, t:({:.2f}, {:.2f}, {:.2f}),' \
               '({:.2f}, {:.2f}, {:.2f}), {:.3f})'.format(
                   self.frame, self.id, self.cls, *self.t, self.l, self.w, self.h, self.ry)


def get_tracklets(label_path):
    """Reads in label data file from Kitti Dataset
    Args:
        label_dir: label directory
        sample_name: sample_name
        results: whether this is a result file
    Returns:
        obj_list: list of ObjectLabels
    """

    # Check label file
    # label_path = label_dir + '/{}.txt'.format(sequence_id)

    if not os.path.exists(label_path):
        raise FileNotFoundError('Label file could not be found')
    if os.stat(label_path).st_size == 0:
        return []

    # TODO: Read into a separate list for each frame?
    labels = np.loadtxt(label_path, delimiter=' ', dtype=str).reshape(-1, 17)

    num_labels = labels.shape[0]
    tracklets = []
    for track_idx in np.arange(num_labels):
        tracklet = Tracklet()

        if labels[track_idx, 2] == 'DontCare':
            continue

        # Fill in the object list
        tracklet.frame = int(labels[track_idx, 0])
        tracklet.id = int(labels[track_idx, 1])
        tracklet.cls = labels[track_idx, 2]
        tracklet.trunc = float(labels[track_idx, 3])
        tracklet.occ = float(labels[track_idx, 4])
        tracklet.alpha = float(labels[track_idx, 5])

        tracklet.x1, tracklet.y1, tracklet.x2, tracklet.y2 = (
            labels[track_idx, 6:10]).astype(np.float32)
        tracklet.h, tracklet.w, tracklet.l = (labels[track_idx, 10:13]).astype(np.float32)
        tracklet.t = (labels[track_idx, 13:16]).astype(np.float32)
        tracklet.ry = float(labels[track_idx, 16])

        tracklets.append(tracklet)

    return np.asarray(tracklets)


def tracklet_to_obj_label(tracklet):
    obj_label = ObjectLabel()
    obj_label.type = tracklet.cls

    obj_label.h = tracklet.h
    obj_label.w = tracklet.w
    obj_label.l = tracklet.l
    obj_label.t = tracklet.t
    obj_label.ry = tracklet.ry

    return obj_label

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import h5py
import json_tricks as json
from tqdm import tqdm
from collections import OrderedDict

import torch
import numpy as np
from scipy.io import loadmat, savemat
from poseutils.constants import *

from dataset.JointsDataset import JointsDataset


logger = logging.getLogger(__name__)


'''
keypoints: {
    0: 'Hip',
    1: 'RHip',
    2: 'RKnee',
    3: 'RAnkle',
    4: 'LHip',
    5: 'LKnee',
    6: 'LAnkle',
    7: 'Neck',
    8: 'LUpperArm',
    9: 'LElbow',
    10: 'LWrist',
    11: 'RUpperArm',
    12: 'RElbow',
    13: 'RWrist'
}
edges: [[0, 1], [0, 4], [0, 7], [1, 2], [2, 3], [4, 5], [5, 6], [7, 8], [8, 9], [9, 10], [7, 11], [11, 12], [12, 13]]
'''


class H36MDataset(JointsDataset):
    def __init__(self, cfg, root, image_set, is_train, transform=None):
        super().__init__(cfg, root, image_set, is_train, transform)

        self.num_joints = 14
        self.flip_pairs = [[1, 4],[2, 5],[3, 6],[11, 8],[12, 9],[13, 10]]
        self.parent_ids = None
        self.aspect_ratio = 1.0
        self.pixel_std = 200

        self.upper_body_ids = (7, 8, 9, 10, 11, 12, 13)
        self.lower_body_ids = (0, 1, 2, 3, 4, 5, 6)

        self.db = self._get_db()

        if is_train and cfg.DATASET.SELECT_DATA:
            self.db = self.select_data(self.db)

        logger.info('=> load {} samples'.format(len(self.db)))

    def _get_db(self):
        # create train/val split

        file_name = os.path.join(self.root, f"meta_{self.image_set}.h5")
        image_dir = os.path.join(self.root, "images", self.image_set)

        gt_db = []

        with h5py.File(file_name) as file_:

            key_count = len(list(file_.keys()))

            for _, key in tqdm(enumerate(file_), total=key_count):
                entry_2d = file_[key]["cropped_2d"][:]
                entry_imgs = file_[key]["image_file_names"][:]
                entry_bins = file_[key]["bins"][:]
                total_frames = len(entry_imgs)

                for i_frame in range(total_frames):

                    center, scale = self._box2cs((0, 0, 256, 256))

                    gt_db.append({
                        "image": os.path.join(image_dir, entry_imgs[i_frame].decode("utf-8")),
                        "joints_3d": entry_2d[i_frame],
                        "center": center,
                        "scale": scale,
                        "joints_3d_vis": np.ones_like(entry_2d[i_frame]),
                        "bins": entry_bins[i_frame]
                    })

            file_.close()

        print(f"{len(gt_db)} data points loaded.")

        return gt_db

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array(
            [w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std],
            dtype=np.float32)
        if center[0] != -1:
            scale = scale * 1.25

        return center, scale

    def __getitem__(self, idx):
        input, target, target_weight, meta = super().__getitem__(idx)

        return input, target, target_weight, meta, torch.from_numpy(self.db[idx]["bins"])

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        # convert 0-based index to 1-based index
        preds = preds[:, :, 0:2] + 1.0

        if output_dir:
            pred_file = os.path.join(output_dir, 'pred.mat')
            savemat(pred_file, mdict={'preds': preds})

        if 'test' in cfg.DATASET.TEST_SET:
            return {'Null': 0.0}, 0.0

        SC_BIAS = 0.6
        threshold = 0.5

        jnt_missing = np.zeros((self.num_joints, preds.shape[0]))
        pos_gt_src = np.vstack([self.db[i]["joints_3d"].reshape(-1, self.num_joints, 2) for i in range(len(self.db))])
        pos_gt_src = pos_gt_src.transpose(1, 2, 0)

        pos_pred_src = np.transpose(preds, [1, 2, 0])

        head = 7
        lsho = 8
        lelb = 9
        lwri = 10
        lhip = 4
        lkne = 5
        lank = 6

        rsho = 11
        relb = 12
        rwri = 13
        rkne = 2
        rank = 3
        rhip = 0

        jnt_visible = 1 - jnt_missing
        uv_error = pos_pred_src - pos_gt_src
        uv_err = np.linalg.norm(uv_error, axis=1)
        scale = np.ones((len(uv_err), 1))
        scaled_uv_err = np.divide(uv_err, scale)
        scaled_uv_err = np.multiply(scaled_uv_err, jnt_visible)
        jnt_count = np.sum(jnt_visible, axis=1)
        less_than_threshold = np.multiply((scaled_uv_err <= threshold),
                                          jnt_visible)
        PCKh = np.divide(100.*np.sum(less_than_threshold, axis=1), jnt_count)

        # save
        rng = np.arange(0, 0.5+0.01, 0.01)
        pckAll = np.zeros((len(rng), self.num_joints))

        for r in range(len(rng)):
            threshold = rng[r]
            less_than_threshold = np.multiply(scaled_uv_err <= threshold,
                                              jnt_visible)
            pckAll[r, :] = np.divide(100.*np.sum(less_than_threshold, axis=1),
                                     jnt_count)

        PCKh = np.ma.array(PCKh, mask=False)
        PCKh.mask[6:8] = True

        jnt_count = np.ma.array(jnt_count, mask=False)
        jnt_count.mask[6:8] = True
        jnt_ratio = jnt_count / np.sum(jnt_count).astype(np.float64)

        name_value = [
            ('Head', PCKh[head]),
            ('Shoulder', 0.5 * (PCKh[lsho] + PCKh[rsho])),
            ('Elbow', 0.5 * (PCKh[lelb] + PCKh[relb])),
            ('Wrist', 0.5 * (PCKh[lwri] + PCKh[rwri])),
            ('Hip', 0.5 * (PCKh[lhip] + PCKh[rhip])),
            ('Knee', 0.5 * (PCKh[lkne] + PCKh[rkne])),
            ('Ankle', 0.5 * (PCKh[lank] + PCKh[rank])),
            ('Mean', np.sum(PCKh * jnt_ratio)),
            ('Mean@0.1', np.sum(pckAll[11, :] * jnt_ratio))
        ]
        name_value = OrderedDict(name_value)

        return name_value, name_value['Mean']

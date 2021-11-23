import warnings
import logging
import itertools
from collections import OrderedDict
from .coco import CocoDataset
import numpy as np
# import os.path as osp

import mmcv
from mmcv.utils import print_log
from terminaltables import AsciiTable
from mmdet.core import eval_recalls
from .api_wrappers import COCO, COCOeval
from .builder import DATASETS
from typing import List

import os
from DOTA_devkit.dota_evaluation import coco_eval
import json
import shutil

from mmcv.parallel import DataContainer as DC
# from .utils import to_tensor, random_scale

@DATASETS.register_module()
class DOTADataset(CocoDataset):
    # Note! same with DOTA2_v2
    CLASSES = ('plane', 'baseball-diamond',
               'bridge', 'ground-track-field',
               'small-vehicle', 'large-vehicle',
               'ship', 'tennis-court',
               'basketball-court', 'storage-tank',
               'soccer-ball-field', 'roundabout',
               'harbor', 'swimming-pool',
               'helicopter')
    pi_degree = 3.1415926535898 / 180.0
    def evaluate(self,
                 results,
                 metric='segm',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=True,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=[0.5],
                 metric_items=None):
        eval_results = self.evaluateCOCO(results, iou_thrs)
        return eval_results

    def prepare_data__(self, detection, imagenames, catnames, keep_ext=False):
        result = dict(zip(catnames, [{'name': [], 'confidence': [], 'detection': []} for _ in range(len(catnames))]))
        for i, d in enumerate(detection):
            if keep_ext:
                image_name = imagenames[i]
            else:
                image_name = os.path.splitext(imagenames[i])[0]
            if d[0].shape[1] == 9:
                for j, _d in enumerate(d):
                    cat_name = catnames[j]
                    if _d.shape[0] == 0:
                        continue
                    else:
                        for line_ in _d.tolist():
                            result[cat_name]['name'].append(image_name)
                            result[cat_name]['confidence'].append(line_[8])
                            result[cat_name]['detection'].append(line_[:-1])
        return result

    def evaluateCOCO(self, results, iou_thrs: List[float]):
        # results = load_result(result_file)
        if isinstance(results, list):
            if isinstance(results[0], np.ndarray):
                import pickle
                with open('/workspace/evaluate_result.pkl', 'wb') as f:
                    pickle.dump(results, f)
        else:
            import pickle
            with open('/workspace/evaluate_result.pkl', 'wb') as f:
                pickle.dump(results, f)

        if iou_thrs is None:
            iou_thrs = [0.5]
        map = [0.0 for _ in iou_thrs]
        classaps = [[] for _ in iou_thrs]
        imagenames = [line['file_name'] for line in self.coco.dataset['images']]
        catnames = [line['name'] for line in self.coco.dataset['categories']]
        d = self.prepare_data__(results, imagenames, catnames, False)

        # map = 0.0
        # classaps = []
        names = []
        for classname in catnames:
            names.append(classname)
            # print('classname:', classname)
            for i, ovthresh in enumerate(iou_thrs):
                rec, prec, ap = coco_eval(d,
                                          self.coco,
                                          classname,
                                          ovthresh=ovthresh,
                                          use_07_metric=True)
                map[i] = map[i] + ap
                # print('rec: ', rec, 'prec: ', prec, 'ap: ', ap)
                # print('ap: ', ap)
                classaps[i].append(ap)
                # umcomment to show p-r curve of each category
                # plt.figure(figsize=(8,4))
                # plt.xlabel('recall')
                # plt.ylabel('precision')
                # plt.plot(rec, prec)
        # plt.show()
        map = [100.0 * m / len(catnames) for m in map]
        # print('map:', map)
        classaps = 100 * np.array(classaps)
        # print('classaps: ', classaps)
        result_json = {}
        for i, ovthresh in enumerate(iou_thrs):
            result_json[f'iou_{ovthresh*100:.0f}'] = dict(mAp=map[i], detail=dict(zip(names, classaps[i])))
        return result_json

@DATASETS.register_module()
class DOTADataset_15(DOTADataset):
    # Note! same with DOTA2_v2
    CLASSES = ('plane', 'baseball-diamond',
               'bridge', 'ground-track-field',
               'small-vehicle', 'large-vehicle',
               'ship', 'tennis-court',
               'basketball-court', 'storage-tank',
               'soccer-ball-field', 'roundabout',
               'harbor', 'swimming-pool',
               'helicopter', 'container-crane')

@DATASETS.register_module()
class DOTADataset_10(DOTADataset):
    CLASSES = ('plane', 'baseball-diamond',
               'bridge', 'ground-track-field',
               'small-vehicle', 'large-vehicle',
               'ship', 'tennis-court',
               'basketball-court', 'storage-tank',
               'soccer-ball-field', 'roundabout',
               'harbor', 'swimming-pool',
               'helicopter')
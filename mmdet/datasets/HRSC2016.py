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

from .DOTA import DOTADataset

@DATASETS.register_module()
class HRSC2016Dataset(DOTADataset):
    CLASSES = ('ship',)
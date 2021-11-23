import numpy as np
import torch

from mmdet.core import PolygonMasks
from mmcv.parallel import DataContainer
from ..builder import PIPELINES
from mmcv.parallel import DataContainer as DC
from .formating import to_tensor
import cv2
from .rotate_base import (RotatePolyRbox, FlipPolyRbox,
                          PolyTransfer, draw_poly,
                          RandomPolyRboxCrop, RandomPolyRboxScale,
                          PolyResize)

@PIPELINES.register_module()
class MaskToRbbox:
    def __init__(self, key='gt_rbboxes', remove_keys=['gt_masks', 'gt_bboxes'], min_size=4):
        self.pi_degree = 3.14159265358979 / 180.0
        self.key = key
        self.remove_keys = remove_keys
        self.min_size = min_size
    def __call__(self, results):
        all_mask = results['gt_masks'].data.masks
        scale_factor = np.array(results['scale_factor'].tolist() + [1.0], dtype=np.float32)
        results['scale_factor'] = scale_factor
        labels = []
        rbboxes = []
        if len(all_mask) > 0:
            all_label = results['gt_labels'].data.numpy().tolist()
            for i in range(all_mask.shape[0]):
                try:
                    conters = cv2.findContours(all_mask[i, ...], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0].squeeze(axis=1)
                except:
                    continue
                (cx, cy), (w, h), angle = cv2.minAreaRect(conters)
                min_size = min(w, h)
                if min_size < self.min_size:
                    continue
                if not angle % 90.0 == angle:
                    h, w = w, h
                    angle %= 90.0
                rbboxes.append([cx, cy, w, h, angle * self.pi_degree])
                labels.append(all_label[i])
        results[self.key] = DataContainer(torch.tensor(rbboxes, dtype=torch.float32))
        results['gt_labels'] = DataContainer(torch.tensor(labels, dtype=torch.int64))
        keys = list(results.keys())
        for key in keys:
            if key in self.remove_keys:
                _ = results.pop(key)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str

@PIPELINES.register_module()
class RandomRotateRbox:
    def __init__(self, *args, **kwargs):
        self.rotater = RotatePolyRbox(*args, **kwargs)

    def __call__(self, results):
        img = results['img']
        n_obj = len(results['gt_labels'])
        if n_obj > 0:
            all_poly = np.stack([poly[0] for poly in results['gt_masks'].masks], axis=0)
            rotated_image, rotated_poly, _ = self.rotater(img, all_poly)
            # image = draw_poly(rotated_image, rotated_poly, results['gt_labels'])
            # cv2.imwrite('/workspace/TEST/test.jpg', image)
            h, w = rotated_image.shape[:2]
            results['rotate_angle'] = self.rotater.get_angle()
            results['img'] = rotated_image
            results['gt_masks'] = PolygonMasks([[p.reshape(-1)] for p in np.split(rotated_poly, n_obj, 0)], h, w)
        else:
            rotated_image, _, _ = self.rotater(img, None)
            results['rotate_angle'] = self.rotater.get_angle()
            results['img'] = rotated_image

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str

@PIPELINES.register_module()
class RandomFlipPoly:
    def __init__(self, *args, **kwargs):
        self.fliper = FlipPolyRbox(*args, **kwargs)

    def __call__(self, results):
        img = results['img']
        n_obj = len(results['gt_labels'])
        if n_obj > 0:
            all_poly = np.stack([poly[0] for poly in results['gt_masks'].masks], axis=0)
            flip_image, flip_poly, _ = self.fliper(img, all_poly)
            h, w = flip_image.shape[:2]
            results['img'] = flip_image
            results['gt_masks'] = PolygonMasks([[p.reshape(-1)] for p in np.split(flip_poly, n_obj, 0)], h, w)
        else:
            flip_image, _, _ = self.fliper(img, None)
            results['img'] = flip_image

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str

@PIPELINES.register_module()
class Poly2Rbox:
    def __init__(self, key='gt_rboxes', *args, **kwargs):
        self.transformer = PolyTransfer(*args, **kwargs)
        self.key = key

    def __call__(self, results):
        n_obj = len(results['gt_labels'])
        if n_obj > 0:
            all_poly = np.stack([poly[0] for poly in results['gt_masks'].masks], axis=0)
            keep_rbox, kepp_label = self.transformer(all_poly, results['gt_labels'])
            results['gt_labels'] = kepp_label
            results[self.key] = keep_rbox
        else:
            results[self.key] = np.empty((0, 5), dtype=np.float32)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str

@PIPELINES.register_module()
class RandomPolyScale:
    def __init__(self, min_size):
        self.min_size = min_size
        self.scaler = RandomPolyRboxScale()

    def __call__(self, image, polys, scale_range):
        scale_image, scale_poly = self.scaler(scale_range, self.min_size, image, polys)
        return scale_image, scale_poly

@PIPELINES.register_module()
class RandomPolyCrop:
    def __init__(self, fix_size, keep_threshold=0.3):
        self.croper = RandomPolyRboxCrop(fix_size, keep_threshold)

    def __call__(self, image, polys, labels):
        crop_image, crop_polys, crop_labels = self.croper(image, polys, labels)
        return crop_image, crop_polys, crop_labels

@PIPELINES.register_module()
class CustomFormatBundle:
    def __init__(self, formatting_keywords=['img', 'gt_rboxes', 'gt_labels']):
        self.formatting_keywords = formatting_keywords

    def __call__(self, results):
        for key in self.formatting_keywords:
            if key in results:
                if key == 'img':
                    img = results['img']
                    if len(img.shape) < 3:
                        img = np.expand_dims(img, -1)
                    img = np.ascontiguousarray(img.transpose(2, 0, 1))
                    results['img'] = DC(to_tensor(img), stack=True)
                elif key == 'gt_masks':
                    results['gt_masks'] = DC(results['gt_masks'], cpu_only=True)
                elif key == 'gt_semantic_seg':
                    results['gt_semantic_seg'] = DC(
                        to_tensor(results['gt_semantic_seg'][None, ...]), stack=True)
                else:
                    if results[key] is None:
                        results[key] = []
                    results[key] = DC(to_tensor(results[key]))
        return results

    def __repr__(self):
        return self.__class__.__name__

@PIPELINES.register_module()
class PolyResizeWithPad:
    def __init__(self, *args, **kwargs):
        self.worker = PolyResize(*args, **kwargs)

    def __call__(self, results):
        img = results['img']
        all_poly = np.stack([poly[0] for poly in results['gt_masks'].masks], axis=0)
        img, polys = self.worker(img, all_poly)
        results['img'] = img
        h, w = img.shape[:2]
        results['gt_masks'] = PolygonMasks([[p.reshape(-1)] for p in np.split(polys, len(polys), 0)], h, w)
        return results

    def __repr__(self):
        return self.__class__.__name__
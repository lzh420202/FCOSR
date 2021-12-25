from __future__ import division
import warnings
import torch
import torch.nn as nn
from mmdet.core import RotBox2Polys, polygonToRotRectangle_batch
from mmdet.core import (bbox_mapping, merge_aug_proposals, merge_aug_bboxes,
                        merge_aug_masks, multiclass_nms, multiclass_nms_rbbox)
from mmdet.core import (build_assigner, bbox2roi, dbbox2roi, bbox2result, build_sampler,
                        dbbox2result, merge_aug_masks, roi2droi, mask2poly,
                        get_best_begin_point, polygonToRotRectangle_batch,
                        gt_mask_bp_obbs_list, choose_best_match_batch,
                        choose_best_Rroi_batch, dbbox_rotate_mapping, bbox_rotate_mapping)

from ..detectors.base import BaseDetector
from ..builder import DETECTORS, build_backbone, build_head, build_neck
import logging


@DETECTORS.register_module()
class FCOSR(BaseDetector):

    def __init__(self,
                 backbone,
                 neck=None,
                 rbbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        assert backbone['type'] in ['ReResNet', 'ResNet', 'ResNeXt', 'Res2Net', 'ResNeSt', 'RegNet',
                                    'CSPDarknet', 'MobileNetV2', 'ShuffleNetV2_Plus', 'MobileNetV2_N']
        assert neck['type'] in ['ReFPN', 'FPN', 'PAFPN', 'RFP']
        super(FCOSR, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)

        if rbbox_head is not None:
            self.rbbox_head = build_head(rbbox_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    @property
    def with_rbbox(self):
        """bool: whether the detector has a rbbox head"""
        return hasattr(self, 'rbbox_head') and self.rbbox_head is not None

    def extract_feat(self, img):
        if isinstance(img, list):
            img = torch.stack(img)
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.rbbox_head(x)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_rboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_bboxes=None,
                      proposals=None,
                      **kwargs):
        x = self.extract_feat(img)

        losses = dict()
        if self.with_rbbox:
            outs = self.rbbox_head(x)
            loss_inputs = outs + (gt_rboxes, gt_labels, self.train_cfg, gt_bboxes)
            losses = self.rbbox_head.loss(*loss_inputs)

        return losses

    def simple_test(self, img, img_meta, proposals=None, rescale=False):
        # 旋转增广测试部分 start
        rotate_test_cfg = self.test_cfg.get('rotate_test', dict(enable=False))
        assert isinstance(rotate_test_cfg, dict)
        if rotate_test_cfg.get('enable'):
            """rot90, 第二个参数表示旋转次数，正数表示逆时针转，负数表示顺时针转。
            旋转检测需要图像的batchsize等于1，且输出给模型的tensor会扩展成0、逆90、180、顺90度的tensor，
            此时的batchsize=4.
            """
            assert len(img) == 1
            rotate_test_num = rotate_test_cfg.get('rot90')
            if rotate_test_num is None:
                rotate_test_num = [0, 1, 2, 3]
            assert isinstance(rotate_test_num, list)
            imgs = []
            for num in rotate_test_num:
                if num == 0:
                    imgs.append(img)
                else:
                    imgs.append(torch.rot90(img, num, [2, 3]))
            img = torch.cat(imgs, dim=0)
            img_meta *= 4
        # 旋转增广测试部分 end
        x = self.extract_feat(img)
        if self.with_rbbox:
            outs = self.rbbox_head(x)
            bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
            bbox_list = self.rbbox_head.get_rbboxes(*bbox_inputs)

            rbbox_results = [
                dbbox2result(det_bboxes, det_labels, self.rbbox_head.num_classes)
                for det_bboxes, det_labels in bbox_list
            ]
        else:
            raise ValueError('must have at least one head.')

        return rbbox_results

    def aug_test(self, imgs, img_metas, rescale=None):
        raise NotImplementedError

    def onnx_export(self, img, img_metas):
        """Test function without nms.

                Args:
                    img (torch.Tensor): input images.
                    img_metas (list[dict]): List of image information.

                Returns:
                    tuple[Tensor, Tensor]: dets of shape [N, num_det, 5]
                        and class labels of shape [N, num_det, class].
                """
        x = self.extract_feat(img)
        outs = self.rbbox_head.forward_onnx(x)
        img_shape = torch._shape_as_tensor(img)[2:]
        img_metas[0]['img_shape_for_onnx'] = img_shape
        # get pad input shape to support onnx dynamic shape for exporting
        # `CornerNet` and `CentripetalNet`, which 'pad_shape' is used
        # for inference
        img_metas[0]['pad_shape_for_onnx'] = img_shape
        bbox_inputs = outs + (img_metas, self.test_cfg)
        det_bboxes, det_labels = self.rbbox_head.get_rbboxes_onnx(*bbox_inputs)

        return det_bboxes, det_labels
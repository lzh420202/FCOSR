import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init, Scale, ConvModule, bias_init_with_prob
from mmcv.ops.focal_loss import sigmoid_focal_loss

from mmdet.core import (multi_apply, multiclass_nms_rbbox, multiclass_poly_nms_rbbox, poly_nms_rbbox,
                        multiclass_poly_nms_rbbox_keep_score, poly_nms_rbbox_keep_score)
from ..losses.gfocal_loss import quality_focal_loss
from ..builder import HEADS, build_loss
from mmdet.ops.fcosr_tools import fcosr_tools
import numpy as np
import math
from mmcv.runner import BaseModule
from ..utils.onnx_utils import (fmod, obbox2corners)

INF = 1e8
PI = 3.14159265359
eps = 1e-9

@HEADS.register_module()
class FCOSRboxHead(BaseModule):
    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 stacked_convs=4,
                 strides=(4, 8, 16, 32, 64),
                 regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                                 (512, INF)),
                 use_sim_ota=False,
                 conv_cfg=None,
                 dcn_on_last_conv=True,
                 drop_positive_sample=dict(enable=False, mode='global', iou_threshold=0.5, keep_min=1),
                 gauss_factor=12.0,
                 image_size=(1024, 1024),
                 loss_cfg=dict(regress=[dict(type='LMious_Loss_v2', k=10, step=0.25, expand=10.0, loss_weight=1.0),
                                        dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)],
                               classify=dict(type='QualityFocalLoss', use_sigmoid=True, beta=2.0, reduction='mean', loss_weight=1.0),
                               classify_score=dict(type='gauss'),
                               regress_weight=dict(type='iou')),
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='fcos_cls',
                         std=0.01,
                         bias_prob=0.01))):
        super(FCOSRboxHead, self).__init__(init_cfg)

        self.num_classes = num_classes
        self.cls_out_channels = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.regress_ranges = regress_ranges
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.image_size = image_size
        self.half_pi = 0.5 * PI
        self.block_size = [int(math.ceil(self.image_size[0]/stride)) * int(math.ceil(self.image_size[1]/stride))
                           for stride in self.strides]
        self.bg_label = num_classes
        self.use_sim_ota = use_sim_ota
        self.sim_ota_topk = 10
        self.sim_ota_reg_factor = 3.0
        self.dcn_on_last_conv = dcn_on_last_conv

        self.use_drop_ps = drop_positive_sample.get('enable', False)
        self.drop_mode = drop_positive_sample.get('mode', 'global')
        self.drop_iou = drop_positive_sample.get('iou_threshold', 0.5)
        self.drop_keep = drop_positive_sample.get('keep_min', 1)
        assert self.drop_mode in ['global', 'local']
        assert self.drop_iou > 0 and self.drop_iou < 1
        assert isinstance(self.drop_keep, int) and (self.drop_keep > 0)

        assert isinstance(gauss_factor, float)
        self.gauss_factor = gauss_factor

        self.use_vfl = False
        class_cfg = loss_cfg.get('classify', None)
        if class_cfg is None:
            self.use_qfl = False
            self.cls_loss_function = build_loss(
                dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0))
        elif isinstance(class_cfg, dict):
            classify_type = class_cfg.get('type')
            if classify_type == 'QualityFocalLoss':
                self.use_qfl = True
            elif classify_type == 'FocalLoss':
                self.use_qfl = False
            elif classify_type == 'VarifocalLoss':
                self.use_qfl = False
                self.use_vfl = True
            else:
                raise ValueError('Unknown classify function!')
            self.cls_loss_function = build_loss(class_cfg)
        else:
            raise ValueError('Unsupport class cfg format.')


        if loss_cfg.get('regress') is None:
            self.reg_loss_function = [build_loss(
                dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))]
        else:
            self.reg_loss_function = [build_loss(cfg) for cfg in loss_cfg['regress']]

        if loss_cfg.get('classify_score') is None:
            self.classify_score = dict(type='gauss')
        else:
            self.classify_score = loss_cfg['classify_score']

        if loss_cfg.get('regress_weight') is None:
            self.regress_weight = dict(type='mean')
        else:
            self.regress_weight = loss_cfg['regress_weight']

        self._init_layers()

        # if torch.onnx.is_in_onnx_export():
        # self.points = self.get_points_onnx([image_size[1], image_size[0]], torch.float32)

    def _init_layers(self):
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            if self.dcn_on_last_conv and i == self.stacked_convs - 1:
                conv_cfg = dict(type='DCNv2')
            else:
                conv_cfg = self.conv_cfg
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))
        self.fcos_cls = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1)
        self.fcos_xy_reg = nn.Conv2d(self.feat_channels, 2, 3, padding=1)
        self.fcos_wh_reg = nn.Conv2d(self.feat_channels, 2, 3, padding=1)
        self.fcos_angle_reg = nn.Conv2d(self.feat_channels, 1, 3, padding=1)

        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

    def forward(self, feats):
        # valide_feats = [feat for k, feat in enumerate(feats) if k < len(self.strides)]
        return multi_apply(self.forward_single, feats, self.scales, self.strides)

    def forward_single(self, x, scale, stride):
        cls_feat = x
        reg_feat = x

        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)
        cls_score = self.fcos_cls(cls_feat)

        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_feat)
        # scale the rbox_pred of different level
        rbox_pred_xy = scale(self.fcos_xy_reg(reg_feat)) * stride
        rbox_pred_wh = (F.elu(scale(self.fcos_wh_reg(reg_feat))) + 1.0) * stride
        rbox_pred_angle = self.fcos_angle_reg(reg_feat).fmod(self.half_pi)

        rbox_pred = torch.cat([rbox_pred_xy, rbox_pred_wh, rbox_pred_angle], 1)

        return cls_score, rbox_pred

    def loss(self,
             cls_scores,
             rbox_preds,
             gt_rboxes,
             gt_labels,
             cfg,
             gt_bboxes_ignore=None):
        assert len(cls_scores) == len(rbox_preds)
        assert len(self.reg_loss_function) > 0
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points, all_level_strides = self.get_points(featmap_sizes, rbox_preds[0].dtype,
                                                              rbox_preds[0].device)
        labels, rbox_targets, nds_scores, rbox_ids = self.fcos_target(all_level_points, gt_rboxes, gt_labels, all_level_strides)
        device = cls_scores[0].device
        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds
        flatten_cls_scores = [cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels) for cls_score in cls_scores]
        flatten_rbox_preds = [rbox_pred.permute(0, 2, 3, 1).reshape(-1, 5) for rbox_pred in rbox_preds]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_rbox_preds = torch.cat(flatten_rbox_preds)
        flatten_labels = torch.cat(labels)
        flatten_rbox_targets = torch.cat(rbox_targets)
        flatten_nds_scores = torch.cat(nds_scores)
        # repeat points to align with bbox_preds
        flatten_points = torch.cat([points.repeat(num_imgs, 1) for points in all_level_points])
        loss = {}
        if self.use_sim_ota:
            rbox_ids_all = torch.cat(rbox_ids)
            num_gt = rbox_ids_all.max().item()
            if num_gt > 0:
                candidate_rbox_ids = rbox_ids_all[rbox_ids_all > 0]
                candidate_pos_inds = ((flatten_labels >= 0) & (flatten_labels < self.bg_label)).nonzero().reshape(-1)
                # candidate_pos_inds = flatten_labels.nonzero().reshape(-1)
                candidate_pos_rbox_targets = flatten_rbox_targets[candidate_pos_inds]
                candidate_pos_rbox_preds = flatten_rbox_preds[candidate_pos_inds]
                candidate_decode_pred = self.vector2bbox(flatten_points[candidate_pos_inds], candidate_pos_rbox_preds.clone())
                candidate_label = flatten_labels[candidate_pos_inds]
                candidate_cls_scores = flatten_cls_scores[candidate_pos_inds]
                candidate_iou = self.get_iou(candidate_decode_pred, candidate_pos_rbox_targets)
                expand_iou = self.expand_score_by_target(candidate_rbox_ids, candidate_iou, num_gt, 0.0)
                sim_ota_topk = min(self.sim_ota_topk, candidate_rbox_ids.shape[0])
                expand_iou_topk_value, expand_iou_topk_index = expand_iou.topk(sim_ota_topk, dim=0)
                dynamic_k = torch.clamp(expand_iou_topk_value.sum(dim=0).int(), min=1)
                del candidate_decode_pred, expand_iou, expand_iou_topk_index, expand_iou_topk_value
                if self.use_qfl:
                    cost_cls = torch.sum(quality_focal_loss(candidate_cls_scores,
                                                            candidate_label,
                                                            torch.ones_like(candidate_label).float(), reduction='none'), dim=1) / self.cls_out_channels
                else:
                    cost_cls = torch.sum(sigmoid_focal_loss(candidate_cls_scores, candidate_label, 2.0, 0.25, 'none'), dim=1) / self.cls_out_channels
                cost_reg = self.sim_ota_reg_factor * (1.0 - candidate_iou)
                cost = cost_cls + cost_reg
                expand_cost = self.expand_score_by_target(candidate_rbox_ids, cost, num_gt, float(1e8))
                del cost, cost_cls, cost_reg
                _, remain_idx = expand_cost.topk(sim_ota_topk, dim=0, largest=False)
                keep_idx = self.get_keep_sample_mask(dynamic_k, remain_idx, candidate_rbox_ids.shape[0])
                candidate_label[keep_idx == 0] = 0
                flatten_labels[candidate_pos_inds] = candidate_label
                del keep_idx, remain_idx, expand_cost, dynamic_k
                loss['ori'] = torch.autograd.Variable(torch.tensor(candidate_pos_inds.shape[0], dtype=torch.float32)[None])
                loss['remain'] = torch.autograd.Variable(torch.tensor(candidate_label.nonzero().shape[0], dtype=torch.float32)[None])
                loss['num_gt'] = torch.autograd.Variable(torch.tensor(num_gt, dtype=torch.float32)[None])

        if self.use_drop_ps:
            candidate_pos_mask = (flatten_labels >= 0) & (flatten_labels < self.bg_label)
            candidate_pos_inds = candidate_pos_mask.nonzero().reshape(-1)
            if len(candidate_pos_inds) > 0:
                candidate_pos_rbox_targets = flatten_rbox_targets[candidate_pos_inds]
                decode_pred = self.vector2bbox(flatten_points[candidate_pos_inds], flatten_rbox_preds[candidate_pos_inds].clone())
                iou = self.get_iou(decode_pred, candidate_pos_rbox_targets).relu()
                del decode_pred
                rbox_ids_all = torch.cat(rbox_ids)
                num_gt = rbox_ids_all.max().item()
                targets_mask = rbox_ids_all > 0
                if not candidate_pos_mask.equal(targets_mask):
                    raise ValueError("pos_mask != targets_mask")
                candidate_rbox_ids = rbox_ids_all[targets_mask]
                expand_iou = self.expand_score_by_target(candidate_rbox_ids.int(), iou, num_gt, -1.0)
                value, index = expand_iou.topk(self.drop_keep, dim=0)
                keep_n = (value >= 0).sum(dim=0)
                keep_index = self.get_keep_sample_mask(keep_n.int(), index, candidate_rbox_ids.shape[0])
                if self.drop_mode == 'global':
                    thre = iou.mean() * self.drop_iou
                    mask = iou > thre
                elif self.drop_mode == 'local':
                    expand_iou_mask = (expand_iou >= 0).clone()
                    n_target = expand_iou_mask.sum(dim=0)
                    expand_iou_ = self.expand_score_by_target(candidate_rbox_ids.int(), iou, num_gt, 0.0)
                    target_iou_mean = expand_iou_.sum(dim=0) / (n_target.float() + 1e-8)
                    thre = (target_iou_mean * self.drop_iou)[None].expand(iou.shape[0], num_gt)
                    mask_ = expand_iou_mask.float()
                    mask_i = (~expand_iou_mask).float()
                    thre = thre * mask_ - 0.5 * mask_i
                    mask = (expand_iou > thre).int().max(dim=1)[0] > 0
                else:
                    raise ValueError(f"Unsupport sample drop mode: {self.drop_mode}")
                mask[keep_index == 1] = True
                candidate_label = flatten_labels[candidate_pos_inds]
                candidate_label[~mask] = self.bg_label
                flatten_labels[candidate_pos_inds] = candidate_label

        pos_inds = ((flatten_labels >= 0) & (flatten_labels < self.bg_label)).nonzero().reshape(-1)
        num_pos = len(pos_inds)
        if self.use_qfl:
            score = flatten_labels.new_zeros(flatten_labels.shape, dtype=torch.float32)
        if num_pos > 0:
            pos_rbox_preds = flatten_rbox_preds[pos_inds]
            pos_rbox_targets = flatten_rbox_targets[pos_inds]
            decode_pred = self.vector2bbox(flatten_points[pos_inds], pos_rbox_preds)
            iou = self.get_iou(decode_pred, pos_rbox_targets).relu()

            if self.use_qfl:
                if self.classify_score['type'] == 'gauss':
                    score[pos_inds] = flatten_nds_scores[pos_inds]
                elif self.classify_score['type'] == 'iou':
                    score[pos_inds] = iou
                elif self.classify_score['type'] == 'none':
                    score[pos_inds] = 1.0

            if self.regress_weight['type'] == 'iou':
                reg_loss_weight = iou
            elif self.regress_weight['type'] == 'mean':
                reg_loss_weight = torch.ones_like(flatten_labels[pos_inds], dtype=torch.float32)
            elif self.regress_weight['type'] == 'gauss':
                reg_loss_weight = flatten_nds_scores[pos_inds]
            elif self.regress_weight['type'] == 'centerness':
                reg_loss_weight = flatten_nds_scores[pos_inds]
            else:
                raise ValueError("regress_weight must be iou or mean.")
            avg_factor = reg_loss_weight.sum()

            for reg_loss_function in self.reg_loss_function:
                loss_type = reg_loss_function._get_name()
                if loss_type == 'SmoothL1Loss':
                    reg_loss_weight_ = reg_loss_weight.view(-1, 1).repeat(1, 5) / 5.0
                    loss[f"loss_{loss_type}"] = reg_loss_function(decode_pred, pos_rbox_targets, reg_loss_weight_, avg_factor=avg_factor)
                else:
                    loss[f"loss_{loss_type}"] = reg_loss_function(decode_pred, pos_rbox_targets, reg_loss_weight, avg_factor=avg_factor)
            loss['IOU_mean'] = torch.autograd.Variable(iou.mean().detach().clone()[None])
        else:
            empty_loss = torch.autograd.Variable(torch.tensor(0, dtype=torch.float32, device=device)[None])
            for reg_loss_function in self.reg_loss_function:
                loss_type = reg_loss_function._get_name()
                loss[f"loss_{loss_type}"] = empty_loss
            loss['IOU_mean'] = empty_loss
        avg_factor = max(1.0, num_pos)
        if self.use_qfl:
            loss_cls = self.cls_loss_function(flatten_cls_scores, (flatten_labels, score), weight=None, avg_factor=avg_factor)[None]
        else:
            # cls_weight = torch.ones_like(flatten_labels, dtype=torch.float32)
            loss_cls = self.cls_loss_function(flatten_cls_scores, flatten_labels, avg_factor=avg_factor)

        loss[f"loss_{self.cls_loss_function._get_name()}"] = loss_cls
        return loss

    def polyRot90(self, polys:torch.Tensor, src_size, num):
        """
        检测结果旋转90度，用于旋转增广测试。
        :param polys: 检测结果， [N, 8]
        :param src_size: 原始图像大小
        :param num: 旋转次数, 正数表示顺时针旋转， 负数表示逆时针旋转
        :return:
        """
        x = polys[:, 0::2].clone()
        y = polys[:, 1::2].clone()
        w, h = src_size
        real_n = num % 4
        if real_n == 0:
            return polys, (w, h)
        elif real_n == 1:
            rot = polys.clone()
            rot[:, 0::2] = h - y - 1
            rot[:, 1::2] = x
            return rot, (h, w)
        elif real_n == 2:
            rot = polys.clone()
            rot[:, 0::2] = w - x - 1
            rot[:, 1::2] = h - y - 1
            return rot, (w, h)
        elif real_n == 3:
            rot = polys.clone()
            rot[:, 0::2] = y
            rot[:, 1::2] = w - x - 1
            return rot, (h, w)
        else:
            raise ValueError(f"Invalid rotate time: num->{num}, real num ->{real_n}.")

    def get_rbboxes(self,
                   cls_scores,
                   bbox_preds,
                   img_metas,
                   cfg,
                   rescale=None):
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_points, _ = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                                    bbox_preds[0].device)
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [cls_scores[i][img_id].detach() for i in range(num_levels)]
            rbbox_pred_list = [bbox_preds[i][img_id].detach() for i in range(num_levels)]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            if scale_factor.size == 4:
                scale_factor = np.array(scale_factor.tolist() * 2, dtype=np.float32)
            elif scale_factor.size == 1:
                scale_factor = np.array(scale_factor.tolist() * 8, dtype=np.float32)
            det_rbboxes = self.get_rbboxes_single(
                cls_score_list, rbbox_pred_list, mlvl_points, img_shape, scale_factor, cfg, rescale)

            result_list.append(det_rbboxes)
        # 旋转增广测试部分 start
        rotate_test_cfg = cfg.get('rotate_test')
        if rotate_test_cfg is None:
            rotate_test_flag = False
        else:
            if rotate_test_cfg.get('enable'):
                rotate_test_flag = True
            else:
                rotate_test_flag = False
        if rotate_test_flag:
            all_rboxes, all_scores = [], []
            rotate_test_num = rotate_test_cfg.get('rot90')
            if rotate_test_num is None:
                rotate_test_num = [0, 1, 2, 3]
            assert isinstance(rotate_test_num, list)
            for i, num in enumerate(rotate_test_num):
                det_rboxes, _, det_scores = result_list[i]
                det_rboxes = det_rboxes[:, :8]
                if num != 0:
                    h, w = img_metas[0]['img_shape'][:2]
                    det_rboxes, _ = self.polyRot90(det_rboxes, (w, h), num)
                all_rboxes.append(det_rboxes.float())
                all_scores.append(det_scores)
            all_rboxes = torch.cat(all_rboxes, dim=0)
            all_scores = torch.cat(all_scores, dim=0)

            total_nms_cfg = cfg.get('totoal_nms')
            if total_nms_cfg is None:
                total_nms_flag = False
            else:
                if total_nms_cfg.get('enable'):
                    total_nms_flag = True
                else:
                    total_nms_flag = False
            if total_nms_flag:
                final_nms_thre = total_nms_cfg.get('iou_thr', 0.8)
                assert isinstance(final_nms_thre, float)
                det_bboxes, det_labels = poly_nms_rbbox(
                    all_rboxes,
                    all_scores,
                    cfg.score_thr,
                    cfg.nms,
                    cfg.max_per_img,
                    score_factors=None,
                    final_nms_thre=final_nms_thre
                )
            else:
                det_bboxes, det_labels = multiclass_poly_nms_rbbox(
                    all_rboxes,
                    all_scores,
                    cfg.score_thr,
                    cfg.nms,
                    cfg.max_per_img,
                    score_factors=None
                )
            return [(det_bboxes, det_labels)]
        # 旋转增广测试部分 end

        return result_list

    # def get_rbboxes_single(self,
    #                        cls_scores,
    #                        rbbox_preds,
    #                        mlvl_points,
    #                        img_shape,
    #                        scale_factor,
    #                        cfg,
    #                        rescale=False):
    #     assert len(cls_scores) == len(rbbox_preds) == len(mlvl_points)
    #     mlvl_rbboxes = []
    #     mlvl_scores = []
    #     for cls_score, rbbox_pred, points in zip(
    #             cls_scores, rbbox_preds, mlvl_points):
    #         assert cls_score.size()[-2:] == rbbox_pred.size()[-2:]
    #         scores = cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels).sigmoid()
    #
    #         rbbox_pred = rbbox_pred.permute(1, 2, 0).reshape(-1, 5)
    #
    #         corners = self.rotate2corner(self.vector2bbox(points, rbbox_pred.clone()))
    #         index = ((corners[:, 0::2] < img_shape[1]) & (corners[:, 0::2] >= 0)) & \
    #                 ((corners[:, 1::2] < img_shape[1]) & (corners[:, 1::2] >= 0))
    #         remain_index = index.sum(1) > 0
    #         rbbox_pred = rbbox_pred[remain_index, :]
    #         points = points[remain_index, :]
    #         scores = scores[remain_index, :]
    #
    #         nms_pre = cfg.get('nms_pre', -1)
    #         if nms_pre > 0 and scores.shape[0] > nms_pre:
    #             # TODO 2
    #             max_scores, _ = scores.max(dim=1)
    #             _, topk_inds = max_scores.topk(nms_pre)
    #             points = points[topk_inds, :]
    #             rbbox_pred = rbbox_pred[topk_inds, :]
    #             scores = scores[topk_inds, :]
    #         # TODO 1
    #         rbboxes = self.vector2bbox(points, rbbox_pred)
    #         # bboxes = distance2bbox(points, rbbox_pred, max_shape=img_shape)
    #         mlvl_rbboxes.append(rbboxes)
    #         mlvl_scores.append(scores)
    #     mlvl_rbboxes = torch.cat(mlvl_rbboxes)
    #     if rescale:
    #         mlvl_rbboxes /= mlvl_rbboxes.new_tensor(scale_factor)
    #     mlvl_scores = torch.cat(mlvl_scores)
    #     det_bboxes, det_labels = multiclass_nms_rbbox(
    #         mlvl_rbboxes,
    #         mlvl_scores,
    #         cfg.score_thr,
    #         cfg.nms,
    #         cfg.max_per_img,
    #         score_factors=None
    #     )
    #     return det_bboxes, det_labels
    def get_rbboxes_single(self,
                           cls_scores,
                           rbbox_preds,
                           mlvl_points,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False):
        assert len(cls_scores) == len(rbbox_preds) == len(mlvl_points)
        mlvl_polys = []
        mlvl_scores = []
        for cls_score, rbbox_pred, points in zip(
                cls_scores, rbbox_preds, mlvl_points):
            assert cls_score.size()[-2:] == rbbox_pred.size()[-2:]
            scores = cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels).sigmoid()

            rbbox_pred = rbbox_pred.permute(1, 2, 0).reshape(-1, 5)

            min_rbox_size = cfg.get('min_bbox_size', 0)
            if min_rbox_size > 0:
                keep_index = (rbbox_pred[:, 2] >= min_rbox_size) & (rbbox_pred[:, 3] >= min_rbox_size)
                if not keep_index.all():
                    rbbox_pred = rbbox_pred[keep_index]
                    points = points[keep_index]
                    scores = scores[keep_index]

            corners = self.rotate2corner(self.vector2bbox(points, rbbox_pred.clone()))
            index = ((corners[:, 0::2] < img_shape[1]) & (corners[:, 0::2] >= 0)) & \
                    ((corners[:, 1::2] < img_shape[1]) & (corners[:, 1::2] >= 0))
            if cfg.get('clip_result', False):
                _2point_index = index.sum(1) == 2
                if _2point_index.any():
                    corners[_2point_index, :] = self.polygon_cut(corners[_2point_index, :], [img_shape[1], img_shape[0]], 'v2', 0.8)
                    index = ((corners[:, 0::2] < img_shape[1]) & (corners[:, 0::2] >= 0)) & \
                            ((corners[:, 1::2] < img_shape[1]) & (corners[:, 1::2] >= 0))
            remain_index = index.sum(1) > 1
            corners_pred = corners[remain_index, :]
            scores = scores[remain_index, :]

            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                # TODO 2
                max_scores, _ = scores.max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                corners_pred = corners_pred[topk_inds, :]
                scores = scores[topk_inds, :]
            mlvl_polys.append(corners_pred)
            mlvl_scores.append(scores)
        mlvl_polys = torch.cat(mlvl_polys)
        if rescale:
            if not (scale_factor == 1).all():
                mlvl_polys /= mlvl_polys.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        total_nms_cfg = cfg.get('totoal_nms')
        if total_nms_cfg is None:
            total_nms_flag = False
        else:
            if total_nms_cfg.get('enable'):
                total_nms_flag = True
            else:
                total_nms_flag = False
        # 旋转增广测试部分 start
        rotate_test_cfg = cfg.get('rotate_test')
        if rotate_test_cfg is None:
            rotate_test_flag = False
        else:
            if rotate_test_cfg.get('enable'):
                rotate_test_flag = True
            else:
                rotate_test_flag = False
        if rotate_test_flag:
            if total_nms_flag:
                final_nms_thre = total_nms_cfg.get('iou_thr', 0.8)
                assert isinstance(final_nms_thre, float)
                det_bboxes, det_labels, det_scores = poly_nms_rbbox_keep_score(
                    mlvl_polys,
                    mlvl_scores,
                    cfg.score_thr,
                    cfg.nms,
                    cfg.max_per_img,
                    score_factors=None,
                    final_nms_thre=final_nms_thre
                )
            else:
                det_bboxes, det_labels, det_scores = multiclass_poly_nms_rbbox_keep_score(
                    mlvl_polys,
                    mlvl_scores,
                    cfg.score_thr,
                    cfg.nms,
                    cfg.max_per_img,
                    score_factors=None
                )
        else:
            det_scores = None
            if total_nms_flag:
                final_nms_thre = total_nms_cfg.get('iou_thr', 0.8)
                assert isinstance(final_nms_thre, float)
                det_bboxes, det_labels = poly_nms_rbbox(
                    mlvl_polys,
                    mlvl_scores,
                    cfg.score_thr,
                    cfg.nms,
                    cfg.max_per_img,
                    score_factors=None,
                    final_nms_thre=final_nms_thre
                )
            else:
                det_bboxes, det_labels = multiclass_poly_nms_rbbox(
                    mlvl_polys,
                    mlvl_scores,
                    cfg.score_thr,
                    cfg.nms,
                    cfg.max_per_img,
                    score_factors=None
                )
        if det_scores is None:
            return det_bboxes, det_labels
        else:
            return det_bboxes, det_labels, det_scores
        # 旋转增广测试部分 end

    def get_points(self, featmap_sizes, dtype, device):
        """Get points according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            dtype (torch.dtype): Type of points.
            device (torch.device): Device of points.

        Returns:
            tuple: points of each image.
        """
        mlvl_points = []
        mlvl_stride = []
        for i in range(len(featmap_sizes)):
            points, strides = self.get_points_single(featmap_sizes[i], self.strides[i],
                                   dtype, device)
            mlvl_points.append(points)
            mlvl_stride.append(strides)
        return mlvl_points, mlvl_stride

    def get_points_single(self, featmap_size, stride, dtype, device):
        h, w = featmap_size
        x_range = torch.arange(
            0, w * stride, stride, dtype=dtype, device=device)
        y_range = torch.arange(
            0, h * stride, stride, dtype=dtype, device=device)
        y, x = torch.meshgrid(y_range, x_range)
        points = torch.stack(
            (x.reshape(-1).float(), y.reshape(-1).float()), dim=-1) + stride / 2.0
        stirdes = torch.ones((points.shape[0],), dtype=torch.float32, device=device) * stride
        return points, stirdes

    def fcos_target(self, points, gt_bboxes_list, gt_labels_list, all_level_strides):
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)
        concat_strides = torch.cat(all_level_strides, dim=0)
        # get labels and bbox_targets of each image
        labels_list, rbox_targets_list, nds_score_list, rboxes_idx_list = multi_apply(
            self.fcos_target_single,
            gt_bboxes_list,
            gt_labels_list,
            points=concat_points,
            regress_ranges=concat_regress_ranges,
            strides=concat_strides)

        num_gt_per_image = [box_id.max().item() for box_id in rboxes_idx_list]
        cumsum_size = np.array(num_gt_per_image).cumsum().tolist()
        start_p = [0] + cumsum_size[:-1]
        rboxes_idx_list = [id + (id > 0).int() * start for id, start in zip(rboxes_idx_list, start_p)]

        # split to per img, per level
        num_points = [center.size(0) for center in points]
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        nds_score_list = [nds_score.split(num_points, 0) for nds_score in nds_score_list]
        rbox_targets_list = [rbox_targets.split(num_points, 0) for rbox_targets in rbox_targets_list]
        rboxes_idx_list = [rboxes_idx.split(num_points, 0) for rboxes_idx in rboxes_idx_list]

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        concat_lvl_nds_scores = []
        concat_lvl_rbox_ids = []
        for i in range(num_levels):
            concat_lvl_labels.append(torch.cat([labels[i] for labels in labels_list]))
            concat_lvl_bbox_targets.append(torch.cat([rbox_targets[i] for rbox_targets in rbox_targets_list]))
            concat_lvl_nds_scores.append(torch.cat([nds_score[i] for nds_score in nds_score_list]))
            concat_lvl_rbox_ids.append(torch.cat([rboxes_idx[i] for rboxes_idx in rboxes_idx_list]))
        return concat_lvl_labels, concat_lvl_bbox_targets, concat_lvl_nds_scores, concat_lvl_rbox_ids

    def fcos_target_single(self, gt_rboxes: torch.Tensor, gt_labels, points, regress_ranges, strides):
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        if num_gts == 0:
            device = points.device
            return (torch.ones((num_points,), dtype=torch.int64, device=device) * self.bg_label,
                    torch.zeros((num_points, 5), dtype=torch.float32, device=device),
                    torch.zeros((num_points,), dtype=torch.float32, device=device),
                    torch.zeros((num_points,), dtype=torch.int32, device=device))

        gt_bboxes = self.rotate2rect(gt_rboxes)
        regress_ranges = regress_ranges[:, None, :].expand(num_points, num_gts, 2)
        gt_rboxes = gt_rboxes[None].expand(num_points, num_gts, 5)
        bbox_targets = gt_bboxes[None].expand(num_points, num_gts, 4)
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        # inside_gt_rbox_mask = self.get_rotate_inside_mask(xs, ys, gt_rboxes)
        ngds_score = self.get_ngds_score(xs, ys, gt_rboxes, mode='shrink', version='v2')
        gds_score = self.get_gds_score(xs, ys, gt_rboxes, mode='shrink', refined=True)


        inside_gt_rbox_mask, gt_rboxes_idx = self.get_rotate_inside_mask_with_gds(xs, ys, gt_rboxes, 0.23, ngds_score, True)
        inside_regress_range = self.get_inside_balance_regress_mask(xs, ys, bbox_targets, gt_rboxes, regress_ranges, strides, factor=1.0)
        ngds_score_gds = ngds_score

        gt_rboxes_gds = gt_rboxes
        gds_score[inside_gt_rbox_mask == 0] = -1.0
        gds_score[inside_regress_range == 0] = -1.0
        max_gds, max_gds_inds = gds_score.max(dim=1)

        labels_gds = gt_labels[max_gds_inds]
        labels_gds[max_gds < 0.0] = self.bg_label

        gt_rboxes_gds = gt_rboxes_gds[range(num_points), max_gds_inds]
        ngds_score_gds = ngds_score_gds[range(num_points), max_gds_inds]
        ngds_score_gds[max_gds < 0.0] = 0
        gt_rboxes_idx = gt_rboxes_idx[range(num_points), max_gds_inds]
        gt_rboxes_idx[max_gds < 0.0] = 0

        return labels_gds, gt_rboxes_gds, ngds_score_gds, gt_rboxes_idx

    def save_mat(self, data, file):
        from scipy.io import savemat
        savemat(file, data)

    def rotate2rect(self, rboxs):
        if not rboxs.is_contiguous():
            rboxs = rboxs.contiguous()
        return fcosr_tools.rbox2rect(rboxs)

    def rotate2corner(self, rboxs):
        if not rboxs.is_contiguous():
            rboxs = rboxs.contiguous()
        return fcosr_tools.rbox2corner(rboxs)

    def get_areas(self, rboxs: torch.Tensor):
        return rboxs[..., 2] * rboxs[..., 3]

    def get_rotate_inside_mask(self, xs: torch.Tensor, ys: torch.Tensor, rboxes: torch.Tensor) \
            -> torch.Tensor:
        if not rboxes.is_contiguous():
            rboxes = rboxes.contiguous()
        if not xs.is_contiguous():
            xs = xs.contiguous()
        if not ys.is_contiguous():
            ys = ys.contiguous()
        return fcosr_tools.get_inside_mask(xs, ys, rboxes)

    def get_rotate_inside_mask_with_gds(
            self, xs: torch.Tensor, ys: torch.Tensor,
            rboxes: torch.Tensor, threshold: float,
            gds: torch.Tensor, with_obj: bool=False) \
            -> torch.Tensor:
        if not rboxes.is_contiguous():
            rboxes = rboxes.contiguous()
        if not xs.is_contiguous():
            xs = xs.contiguous()
        if not ys.is_contiguous():
            ys = ys.contiguous()
        if not gds.is_contiguous():
            gds = gds.contiguous()
        if with_obj:
            return fcosr_tools.get_inside_mask_with_obj_gds(xs, ys, rboxes, threshold, gds)
        else:
            return fcosr_tools.get_inside_mask_with_gds(xs, ys, rboxes, threshold, gds)

    def get_inside_regress_mask(self, xs: torch.Tensor, ys: torch.Tensor, gt_bboxes: torch.Tensor, regress_ranges: torch.Tensor):
        if not gt_bboxes.is_contiguous():
            gt_bboxes = gt_bboxes.contiguous()
        if not xs.is_contiguous():
            xs = xs.contiguous()
        if not ys.is_contiguous():
            ys = ys.contiguous()
        if not regress_ranges.is_contiguous():
            regress_ranges = regress_ranges.contiguous()

        return fcosr_tools.get_inside_regress_mask(xs, ys, gt_bboxes, regress_ranges)

    def get_inside_balance_regress_mask(
            self, xs: torch.Tensor, ys: torch.Tensor,
            gt_bboxes: torch.Tensor, gt_rboxes: torch.Tensor,
            regress_ranges: torch.Tensor, strides: torch.Tensor, factor: float):
        if not gt_bboxes.is_contiguous():
            gt_bboxes = gt_bboxes.contiguous()
        if not gt_rboxes.is_contiguous():
            gt_rboxes = gt_rboxes.contiguous()
        if not strides.is_contiguous():
            strides = strides.contiguous()
        if not xs.is_contiguous():
            xs = xs.contiguous()
        if not ys.is_contiguous():
            ys = ys.contiguous()
        if not regress_ranges.is_contiguous():
            regress_ranges = regress_ranges.contiguous()

        return fcosr_tools.get_inside_balance_regress_mask(xs, ys, gt_bboxes, gt_rboxes, regress_ranges, strides, factor)

    def get_inside_balance_regress_mask_v2(
            self, xs: torch.Tensor, ys: torch.Tensor,
            gt_rboxes: torch.Tensor, regress_ranges: torch.Tensor,
            strides: torch.Tensor, factor: float):
        if not gt_rboxes.is_contiguous():
            gt_rboxes = gt_rboxes.contiguous()
        if not strides.is_contiguous():
            strides = strides.contiguous()
        if not xs.is_contiguous():
            xs = xs.contiguous()
        if not ys.is_contiguous():
            ys = ys.contiguous()
        if not regress_ranges.is_contiguous():
            regress_ranges = regress_ranges.contiguous()

        return fcosr_tools.get_inside_balance_regress_mask_v2(xs, ys, gt_rboxes, regress_ranges, strides, factor)

    def get_ngds_score(self, xs, ys, gt_rboxes, mode='normal', version='v2'):
        assert mode in ['normal', 'shrink']
        assert version in ['v1', 'v2']
        if mode == 'normal':
            mode_value = 0
        elif mode == 'shrink':
            mode_value = 1
        else:
            raise ValueError
        if not xs.is_contiguous():
            xs = xs.contiguous()
        if not ys.is_contiguous():
            ys = ys.contiguous()
        if not gt_rboxes.is_contiguous():
            gt_rboxes = gt_rboxes.contiguous()
        if version == 'v1':
            return fcosr_tools.get_ngds_score(xs, ys, gt_rboxes, self.gauss_factor, mode_value, self.block_size)
        elif version == 'v2':
            return fcosr_tools.get_ngds_score_v2(xs, ys, gt_rboxes, self.gauss_factor, mode_value)
        else:
            raise ValueError(f'version: {version} is not supported')

    def get_gds_score(self, xs, ys, gt_rboxes, mode='normal', refined=False):
        assert mode in ['normal', 'shrink']
        if mode == 'normal':
            mode_value = 0
        elif mode == 'shrink':
            mode_value = 1
        else:
            raise ValueError
        if not xs.is_contiguous():
            xs = xs.contiguous()
        if not ys.is_contiguous():
            ys = ys.contiguous()
        if not gt_rboxes.is_contiguous():
            gt_rboxes = gt_rboxes.contiguous()
        return fcosr_tools.get_gds_score(xs, ys, gt_rboxes, self.gauss_factor, mode_value, refined)

    def get_iou(self, rboxes_1: torch.Tensor, rboxes_2: torch.Tensor):
        if not rboxes_1.is_contiguous():
            rboxes_1 = rboxes_1.contiguous()
        if not rboxes_2.is_contiguous():
            rboxes_2 = rboxes_2.contiguous()
        # return fcosr_tools.compute_poly_iou(fcosr_tools.rbox2corner(rboxes_1, angle_positive), fcosr_tools.rbox2corner(rboxes_2, angle_positive))
        return fcosr_tools.compute_rbox_iou(rboxes_1, rboxes_2)

    def expand_score_by_target(self, box_ids: torch.Tensor, iou: torch.Tensor, num_gt: int, filled_value: float):
        if not box_ids.is_contiguous():
            box_ids = box_ids.contiguous()
        if not iou.is_contiguous():
            iou = iou.contiguous()
        return fcosr_tools.expand_score(box_ids, iou, num_gt, filled_value)

    def get_keep_sample_mask(self, dynamic_k: torch.Tensor, topk_idx: torch.Tensor, n_sample: int):
        if not dynamic_k.is_contiguous():
            dynamic_k = dynamic_k.contiguous()
        if not topk_idx.is_contiguous():
            topk_idx = topk_idx.contiguous()
        return fcosr_tools.get_keep_sample_idx(dynamic_k, topk_idx, n_sample)

    def vector2bbox(self, points, box_preds):
        box_preds[..., 0:2] = box_preds[..., 0:2] + points
        return box_preds

    def polygon_cut(self, polys: torch.Tensor, image_size, version='v2', keep_threshold=0.8):
        """
        多边形裁剪，使用重写了DOTA_devkit中多边形裁剪的部分，将检测结果限定在图像范围内。
        :param polys: [N, 8]
        :param image_size: List[width, height]
        :param version: str 'v1', 'v2'
        :param keep_threshold: use for version v2
        :return: new polys -> torch.Tensor [N, 8]
        如果选择版本1，若裁减后多边形顶点数大于5个，则被放弃。
        如果选择版本2，若裁减后多边形顶点数大于5个，且裁剪区域面积占原始检测框面积的比例超过保留阈值，则保留检测结果(不裁剪)。
        被放弃目标框所有点坐标值均置为-1.0
        """
        assert isinstance(image_size, list)
        assert len(image_size) == 2
        assert isinstance(image_size[0], int)
        assert isinstance(image_size[1], int)

        assert isinstance(version, str)
        assert version in ['v1', 'v2']

        assert isinstance(keep_threshold, float)
        assert (keep_threshold > 0.0) and (keep_threshold < 1.0)

        if not polys.is_contiguous():
            polys = polys.contiguous()
        if version == 'v1':
            return fcosr_tools.poly_cut(polys, image_size)
        elif version == 'v2':
            return fcosr_tools.poly_cut_v2(polys, image_size, keep_threshold)
        else:
            raise ValueError(f"Unsupport function version:{version}")

    def vector2bbox_onnx(self, points, box_preds):
        # box_preds + torch.constant_pad_nd(points, [0, 3])
        pad_points = torch.cat([points, torch.zeros([points.shape[0], 3], dtype=torch.float32, device=points.device)], dim=1)
        return box_preds + pad_points

    def get_points_onnx(self, image_shape, dtype, device):
        mlvl_points = []
        for i in range(len(self.strides)):
            points = self.get_points_onnx_single(image_shape, self.strides[i], dtype, device)
            mlvl_points.append(points)
        return mlvl_points

    def get_points_onnx_single(self, image_shape, stride, dtype, device):
        x_range = torch.arange(0, image_shape[1], stride, dtype=dtype, device=device)
        y_range = torch.arange(0, image_shape[0], stride, dtype=dtype, device=device)

        y, x = torch.meshgrid(y_range, x_range)
        points = torch.stack((x.reshape(-1).float(), y.reshape(-1).float()), dim=-1) + stride * 0.5
        return points

    def forward_onnx(self, feats):
        return multi_apply(self.forward_onnx_single, feats, self.scales, self.strides)

    def forward_onnx_single(self, x, scale, stride):
        cls_feat = x
        reg_feat = x

        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)
        cls_score = self.fcos_cls(cls_feat)

        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_feat)
        # scale the rbox_pred of different level
        rbox_pred_xy = scale(self.fcos_xy_reg(reg_feat)) * stride
        rbox_pred_wh = (F.elu(scale(self.fcos_wh_reg(reg_feat))) + 1.0) * stride
        # rbox_pred_angle = self.fcos_angle_reg(reg_feat).fmod(self.half_pi)
        # rbox_pred_angle = self.fcos_angle_reg(reg_feat)
        rbox_pred_angle = fmod(self.fcos_angle_reg(reg_feat), self.half_pi)
        rbox_pred = torch.cat([rbox_pred_xy, rbox_pred_wh, rbox_pred_angle], 1)

        return cls_score, rbox_pred

    def get_rbboxes_onnx(self,
                   cls_scores,
                   bbox_preds,
                   img_metas,
                   cfg):
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)
        mlvl_points = self.get_points_onnx(img_metas[0]['img_shape'], torch.float32, bbox_preds[0].device)
        box_lists = []
        label_lists = []
        for img_id in range(len(img_metas)):
            cls_score_list = [cls_scores[i][img_id].detach() for i in range(num_levels)]
            rbbox_pred_list = [bbox_preds[i][img_id].detach() for i in range(num_levels)]
            det_rbboxes, det_labels = self.get_rbboxes_onnx_single(cls_score_list, rbbox_pred_list, mlvl_points, cfg)
            box_lists.append(det_rbboxes)
            label_lists.append(det_labels)

        return torch.stack(box_lists), torch.stack(label_lists)

    def get_rbboxes_onnx_single(self,
                           cls_scores,
                           rbbox_preds,
                           mlvl_points,
                           cfg):
        # assert len(cls_scores) == len(rbbox_preds) == len(mlvl_points)
        mlvl_rbboxes = []
        mlvl_scores = []
        for cls_score, rbbox_pred, points in zip(
                cls_scores, rbbox_preds, mlvl_points):
            # assert cls_score.size()[-2:] == rbbox_pred.size()[-2:]
            scores = cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels).sigmoid()
            rbbox_pred = rbbox_pred.permute(1, 2, 0).reshape(-1, 5)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                # TODO 2
                max_scores, _ = scores.max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                rbbox_pred = rbbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
            # TODO 1
            rbboxes = self.vector2bbox_onnx(points, rbbox_pred)
            # bboxes = distance2bbox(points, rbbox_pred, max_shape=img_shape)
            mlvl_rbboxes.append(obbox2corners(rbboxes))
            # mlvl_rbboxes.append(rbboxes)
            mlvl_scores.append(scores)
        mlvl_rbboxes = torch.cat(mlvl_rbboxes)
        mlvl_scores = torch.cat(mlvl_scores)
        return mlvl_rbboxes, mlvl_scores
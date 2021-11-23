import cv2
import mmcv.image
import numpy as np
from numpy import random
from typing import Tuple
import random
import shapely.geometry as shgeo
import math

def draw_poly(image, boxes, labels=None):
    for i, box in enumerate(boxes):
        pts = np.round(box).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(image, pts=[pts], isClosed=True, color=(0, 0, 255), thickness=2)
        if not labels is None:
            image = cv2.putText(image, str(labels[i]), tuple(pts[0, :, :].reshape(2).tolist()), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255))
    return image

class RotatePolyRbox(object):
    def __init__(self,
                 scale=1.0,
                 border_value=0,
                 auto_bound=True,
                 rotate_range=(-180, 180),
                 rotate_ratio=1.0,
                 rotate_values=[0, 45, 90, 135, 180, -135, -90, -45],
                 rotate_mode='range'):
        self.scale = scale
        self.border_value = border_value
        self.auto_bound = auto_bound
        self.rotate_range = rotate_range
        self.rotate_ratio = rotate_ratio
        self.rotate_values = rotate_values
        self.rotate_mode = rotate_mode
        self.angle = None

    def rotate_box(self, angle, points, src_size, dst_size, auto_scale=True):
        offset_w = (dst_size[0] - src_size[0]) * 0.5
        offset_h = (dst_size[1] - src_size[1]) * 0.5
        center = ((dst_size[0] - 1) * 0.5, (dst_size[1] - 1) * 0.5)
        mat = cv2.getRotationMatrix2D(center, -angle, 1.0)
        mat = np.concatenate([mat, np.array([0.0, 0.0, 1.0]).reshape(1, -1)], axis=0)
        offset_mat = np.array([1.0, 0.0, offset_w, 0.0, 1.0, offset_h, 0.0, 0.0, 1.0]).reshape(3, 3)
        if auto_scale:
            scale_x = np.min(np.array(src_size) / np.array(dst_size))
            scale_mat = np.array([scale_x, 0.0, 0.0, 0.0, scale_x, 0.0, 0.0, 0.0, 1.0]).reshape(3, 3)
        else:
            scale_mat = np.eye(3)
        final_mat = np.matmul(np.matmul(scale_mat, mat), offset_mat)
        num_points = int(points.size / 2)
        rotated_points = np.matmul(final_mat,
                                   np.concatenate([points.reshape(-1, 2).T, np.ones((1, num_points), dtype=np.float64)],
                                                  axis=0))
        return rotated_points[:2, :].T.reshape(-1, 8)

    def get_randon_pad_size(self, image_size: Tuple, pad_unit: int = 32):
        '''
        :param image_size:
        :param pad_unit:
        :return:
        '''
        h, w = image_size
        mod_w = w % pad_unit
        mod_h = h % pad_unit
        pad_size_w = pad_unit - mod_w if mod_w > 0 else 0
        pad_size_h = pad_unit - mod_h if mod_h > 0 else 0
        if pad_size_h > 0:
            list_t = list(range(pad_size_h + 1))
            top = random.choice(list_t)
        else:
            top = 0
        if pad_size_w > 0:
            list_l = list(range(pad_size_w + 1))
            left = random.choice(list_l)
        else:
            left = 0
        bottom = pad_size_h - top
        right = pad_size_w - left
        return (left, right, top, bottom)

    def get_randon_expand_size(self, src_size: Tuple, fix_size: Tuple):
        expand_h = fix_size[0] - src_size[0]
        expand_w = fix_size[1] - src_size[1]
        if expand_h > 0:
            list_t = list(range(expand_h + 1))
            top = random.choice(list_t)
        else:
            top = 0
        if expand_w > 0:
            list_w = list(range(expand_w + 1))
            left = random.choice(list_w)
        else:
            left = 0
        bottom = expand_h - top
        right = expand_w - left
        return (left, right, top, bottom)

    def translation_box(self, l, t, points: np.ndarray):
        new = points.copy()
        new[:, 0::2] += l
        new[:, 1::2] += t
        return new

    def randon_angle(self):
        self.angle = None
        if np.random.rand() > self.rotate_ratio:
            angle = 0.0
        else:
            if self.rotate_mode == 'range':
                angle = np.random.rand() * (self.rotate_range[1] - self.rotate_range[0]) + self.rotate_range[0]
            elif self.rotate_mode == 'value':
                random.shuffle(self.rotate_values)
                angle = self.rotate_values[0]
            else:
                raise ValueError('rotate mode only support range and value.')
        return angle

    def get_angle(self):
        return self.angle

    def rotate_image(self, image, angle=None):
        with_n_size = True
        if angle is None:
            angle = self.randon_angle()
            with_n_size = False
        h, w = image.shape[:2]
        center = ((w - 1) * 0.5, (h - 1) * 0.5)
        matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)
        cos_a = np.abs(matrix[0, 0])
        sin_a = np.abs(matrix[0, 1])
        new_w = h * sin_a + w * cos_a
        new_h = h * cos_a + w * sin_a
        n_w = int(np.round(new_w))
        n_h = int(np.round(new_h))
        if self.auto_bound:
            ratio = min(w/n_w, h/n_h)
            matrix = cv2.getRotationMatrix2D(center, -angle, ratio)
            size_w = w
            size_h = h
        else:
            matrix[0, 2] += (new_w - w) * 0.5
            matrix[1, 2] += (new_h - h) * 0.5
            size_w = n_w
            size_h = n_h
        rotate_image = cv2.warpAffine(image, matrix, (size_w, size_h), borderValue=self.border_value)
        if with_n_size:
            return rotate_image, angle, (n_w, n_h)
        else:
            return rotate_image, angle

    def __call__(self, image, rboxes_8, labels=None):
        h, w = image.shape[:2]
        angle = self.randon_angle()
        rotated_img, angle, n_size = self.rotate_image(image, angle)
        self.angle = angle
        if rboxes_8 is None:
            new_box = None
        else:
            new_box = self.rotate_box(angle, rboxes_8, (w, h), n_size, self.auto_bound)
        return rotated_img, new_box, labels

class FlipPolyRbox(object):
    def __init__(self, mode):
        assert mode in ['vertical', 'horizontal', 'both']
        # 0 none, 1 vertical, 2, horizontal, 3 both
        if mode == 'vertical':
            self.mode_list = [0, 1]
        elif mode == 'horizontal':
            self.mode_list = [0, 2]
        elif mode == 'both':
            self.mode_list = list(range(4))
        else:
            raise ValueError
    @staticmethod
    def flip_points(boxes_8, size_image, dim):
        '''
        :param boxes_8:
        :param size_image: Tuple (w, h)
        :param dim:
        :return:
        '''
        if not boxes_8 is None:
            points = boxes_8.copy().reshape(-1, 2)
            points[:, dim] = size_image[dim] - 1 - points[:, dim]
            points = points.reshape(-1, 8)
        else:
            points = None
        return points

    def flip_image(self, image):
        flip_mode = random.choice(self.mode_list)
        if flip_mode == 0:
            flip_image = image
        elif flip_mode == 1:
            flip_image = cv2.flip(image, 0)
        elif flip_mode == 2:
            flip_image = cv2.flip(image, 1)
        elif flip_mode == 3:
            flip_image = cv2.flip(image, -1)
        else:
            raise ValueError
        return flip_image

    def __call__(self, image, boxes_8, labels=None):
        flip_mode = random.choice(self.mode_list)
        h, w = image.shape[:2]
        if flip_mode == 0:
            flip_image = image
            flip_boxes = boxes_8
            keep_labels = labels
        elif flip_mode == 1:
            # vertical
            flip_image = cv2.flip(image, 0)
            flip_boxes = self.flip_points(boxes_8, (w, h), 1)
            keep_labels = labels
        elif flip_mode == 2:
            # horizontal
            flip_image = cv2.flip(image, 1)
            flip_boxes = self.flip_points(boxes_8, (w, h), 0)
            keep_labels = labels
        elif flip_mode == 3:
            # both
            flip_image = cv2.flip(image, -1)
            flip_boxes = self.flip_points(boxes_8, (w, h), 0)
            flip_boxes = self.flip_points(flip_boxes, (w, h), 1)
            keep_labels = labels
        else:
            raise ValueError
        # print(f"flip_mode: {flip_mode}")
        # image = draw_poly(flip_image, flip_boxes)
        # cv2.imwrite('/workspace/test.jpg', image)
        return flip_image, flip_boxes, keep_labels

class PolyTransfer(object):
    half_pi = 3.14159265359 / 180.0
    def __init__(self, to_rbox=True,
                 filter_cfg=dict(enable=False, mode='edge', threshold=4)):
        self.to_rbox = to_rbox
        self.filter_cfg = filter_cfg

    def filters(self, size):
        flag = False
        if self.filter_cfg['mode'] == 'edge':
            if min(size) < self.filter_cfg['threshold']:
                flag = True
        elif self.filter_cfg['mode'] == 'area':
            if (size[0] * size[1]) < self.filter_cfg['threshold']:
                flag = True
        return flag

    def poly2rbox(self, polys: np.ndarray, labels=None):
        if labels is None:
            self.filter_cfg['enable'] = False
        rbox = []
        keep_label = []
        for i, poly in enumerate(polys):
            points = np.round(poly.reshape(-1, 2)).astype(np.int32)
            (cx, cy), (w, h), angle = cv2.minAreaRect(points)
            if self.filter_cfg['enable']:
                if self.filters((w, h)):
                    continue
            if angle == -0.0:
                angle = 0.0
            if angle == 90.0:
                angle = 0.0
                w, h = h, w
            rbox.append(np.array([cx, cy, w, h, angle * self.half_pi], dtype=np.float32).reshape(1, -1))
            if not labels is None:
                keep_label.append(labels[i])

        if labels is None:
            keep_labels = None
        else:
            keep_labels = np.array(keep_label)
        if len(rbox) == 0:
            keep_rboxes = None
        else:
            keep_rboxes = np.concatenate(rbox, axis=0)
        return keep_rboxes, keep_labels

    def __call__(self, polys, labels=None):
        if self.to_rbox:
            return self.poly2rbox(polys, labels)
        return None, None

class RandomPolyRboxScale(object):
    def randomScale(self, scale_range: Tuple):
        return np.random.rand() * (scale_range[1] - scale_range[0]) + scale_range[0]

    def __call__(self, scale_range, min_size, image, polys):
        min_w, min_h = min_size
        scale = self.randomScale(scale_range)
        h, w = image.shape[:2]
        n_h, n_w = max(scale * h, min_h), max(scale * w, min_w)
        real_scale = max(n_w / w, n_h / h)
        n_h, n_w = max(int(round(real_scale * h)), min_h), max(int(round(real_scale * w)), min_w)
        if h == min_h or w == min_w:
            return image, polys

        scale_mat = np.array([real_scale, 0.0, 0.0, 0.0, real_scale, 0.0, 0.0, 0.0, 1.0]).reshape(3, 3)
        expand_polys = np.concatenate([polys.reshape(-1, 2).T, np.ones((1, polys.shape[0] * 4), dtype=np.int64)], axis=0)
        new_polys = np.matmul(scale_mat, expand_polys)[:2, :].T.reshape(-1, 8)
        new_image = cv2.resize(image, (n_h, n_w))
        return new_image, new_polys

class RandomPolyRboxCrop(object):
    def __init__(self, fix_size, keep_threshold=0.3):
        self.crop_size = fix_size
        self.keep_threshold = keep_threshold

    def calchalf_iou(self, poly1, poly2):
        """
            It is not the iou on usual, the iou is the value of intersection over poly1
        """
        inter_poly = poly1.intersection(poly2)
        inter_area = inter_poly.area
        poly1_area = poly1.area
        half_iou = inter_area / poly1_area
        return inter_poly, half_iou

    def polyorig2sub(self, left, up, poly):
        return [[points[0]-left, points[1]-up] for points in poly]

    def cal_line_length(self, point1, point2):
        return math.sqrt(math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2))

    def GetPoly4FromPoly5(self, poly):
        distances = [self.cal_line_length((poly[i * 2], poly[i * 2 + 1]), (poly[(i + 1) * 2], poly[(i + 1) * 2 + 1])) for i in range(int(len(poly)/2 - 1))]
        distances.append(self.cal_line_length((poly[0], poly[1]), (poly[8], poly[9])))
        pos = np.array(distances).argsort()[0]
        count = 0
        outpoly = []
        while count < 5:
            #print('count:', count)
            if (count == pos):
                outpoly.append((poly[count * 2] + poly[(count * 2 + 2) % 10])/2)
                outpoly.append((poly[(count * 2 + 1) % 10] + poly[(count * 2 + 3) % 10])/2)
                count = count + 1
            elif (count == (pos + 1)%5):
                count = count + 1
                continue

            else:
                outpoly.append(poly[count * 2])
                outpoly.append(poly[count * 2 + 1])
                count = count + 1
        return outpoly

    def choose_best_pointorder_fit_another(self, poly1, poly2):
        """
            To make the two polygons best fit with each point
        """
        x1 = poly1[0]
        y1 = poly1[1]
        x2 = poly1[2]
        y2 = poly1[3]
        x3 = poly1[4]
        y3 = poly1[5]
        x4 = poly1[6]
        y4 = poly1[7]
        combinate = [np.array([x1, y1, x2, y2, x3, y3, x4, y4]), np.array([x2, y2, x3, y3, x4, y4, x1, y1]),
                     np.array([x3, y3, x4, y4, x1, y1, x2, y2]), np.array([x4, y4, x1, y1, x2, y2, x3, y3])]
        dst_coordinate = np.array(poly2).reshape(-1)
        distances = np.array([np.sum((coord - dst_coordinate) ** 2) for coord in combinate])
        sorted = distances.argsort()
        return combinate[sorted[0]]

    def __call__(self, image, rboxes, labels):
        h, w = image.shape[:2]
        c_h, c_w = self.crop_size
        if c_h == h and c_w == w:
            return image, rboxes, labels
        h_range, w_range = h - c_h, w - c_w
        offset_w_list, offset_h_list = list(range(w_range)), list(range(h_range))
        if len(offset_h_list) > 0:
            n_h = random.choice(offset_h_list)
        else:
            n_h = 0
        if len(offset_w_list) > 0:
            n_w = random.choice(offset_w_list)
        else:
            n_w = 0
        crop_image = image[n_h:n_h + self.crop_size[1], n_w:n_w + self.crop_size[0], :]
        crop_ = shgeo.Polygon([[n_w, n_h],
                               [n_w, n_h + self.crop_size[1] - 1],
                               [n_w + self.crop_size[0] - 1, n_h + self.crop_size[1] - 1],
                               [n_w + self.crop_size[0] - 1, n_h]])
        rboxes_list = rboxes.reshape(-1, 4, 2).tolist()
        rboxes_ = [shgeo.Polygon(rbox) for rbox in rboxes_list]
        keep_box = []
        keep_label = []
        for i, poly in enumerate(rboxes_):
            inter_poly, half_iou = self.calchalf_iou(poly, crop_)

            if (half_iou == 1):
                polyInsub = self.polyorig2sub(n_w, n_h, rboxes_list[i])
                keep_box.append(polyInsub)
                keep_label.append(labels[i])
            elif (half_iou > 0):
                if half_iou < self.keep_threshold:
                    continue
                # elif (half_iou > self.thresh):
                ##  print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
                inter_poly = shgeo.polygon.orient(inter_poly, sign=1)
                out_poly = list(inter_poly.exterior.coords)[0: -1]
                if len(out_poly) < 4:
                    continue

                out_poly2 = []
                for i in range(len(out_poly)):
                    out_poly2.append(out_poly[i][0])
                    out_poly2.append(out_poly[i][1])

                if (len(out_poly) == 5):
                    # print('==========================')
                    out_poly2 = self.GetPoly4FromPoly5(out_poly2)
                elif (len(out_poly) > 5):
                    """
                        if the cut instance is a polygon with points more than 5, we do not handle it currently
                    """
                    continue

                out_poly2 = self.choose_best_pointorder_fit_another(out_poly2, rboxes_list[i])
                polyInsub = self.polyorig2sub(n_w, n_h, out_poly2.reshape(-1, 2).tolist())
                polyInsub = [[min(max(point[0], 1), self.crop_size[0]), min(max(point[1], 1), self.crop_size[1])] for point in polyInsub]

                keep_box.append(polyInsub)
                keep_label.append(labels[i])

        return crop_image, np.array(keep_box).reshape(-1, 8), np.array(keep_label)

class PolyResize(object):
    def __init__(self, image_size):
        self.image_size = image_size

    def random_pad(self, image, polys):
        h, w = image.shape[0:2]
        width_flag = False
        height_flag = False
        if self.image_size[0] != w:
            width_flag = True
        if self.image_size[1] != h:
            height_flag = True
        pad_left = 0
        pad_top = 0
        pad_right = 0
        pad_bottom = 0
        if width_flag:
            pad_left = random.choice(range(self.image_size[0] - w + 1))
            pad_right = self.image_size[0] - w - pad_left
            if pad_left > 0:
                polys[:, 0::2] += pad_left
        if height_flag:
            pad_top = random.choice(range(self.image_size[1] - h + 1))
            pad_bottom = self.image_size[0] - h - pad_top
            if pad_top > 0:
                polys[:, 1::2] += pad_top
        image = mmcv.image.impad(image, padding=(pad_left, pad_top, pad_right, pad_bottom), pad_val=0, padding_mode='constant')
        return image, polys

    def __call__(self, image, polys):
        h, w = image.shape[:2]
        if w < h:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            x = polys[:, 0::2].copy()
            y = polys[:, 1::2].copy()
            polys[:, 0::2] = h - y - 1
            polys[:, 1::2] = x
            h, w = image.shape[:2]

        if (h == self.image_size[1] and w <= self.image_size[0]) or \
                (w == self.image_size[0] and h <= self.image_size[1]):
            return self.random_pad(image, polys)
        else:
            new_image, real_scale = mmcv.image.imrescale(image, self.image_size, return_scale=True)
            scale_mat = np.array([real_scale, 0.0, 0.0, 0.0, real_scale, 0.0, 0.0, 0.0, 1.0]).reshape(3, 3)
            expand_polys = np.concatenate([polys.reshape(-1, 2).T, np.ones((1, polys.shape[0] * 4), dtype=np.int64)], axis=0)
            new_polys = np.matmul(scale_mat, expand_polys)[:2, :].T.reshape(-1, 8)
            new_image, new_polys = self.random_pad(new_image, new_polys)
        # image = draw_poly(new_image, new_polys)
        # cv2.imwrite('/workspace/test.jpg', image)
        return new_image, new_polys


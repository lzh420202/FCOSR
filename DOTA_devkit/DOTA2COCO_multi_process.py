import dota_utils as util
import os
import cv2
import json
from multiprocessing import (Queue, Process)
import math
from DOTA2COCO import (wordname_15, wordname_16, wordname_18)


def anno_process_unit(imageparent, input_files, files_ids, cls_names, data_queue, difficult, ext):

    for file, image_id in zip(input_files, files_ids):
        data_dict = dict(annotations=[])
        basename = util.custombasename(file)
        # image_id = int(basename[1:])

        imagepath = os.path.join(imageparent, basename + ext)
        img = cv2.imread(imagepath)
        height, width, c = img.shape

        single_image = {}
        single_image['file_name'] = basename + ext
        single_image['id'] = image_id
        single_image['width'] = width
        single_image['height'] = height
        data_dict['images'] = single_image

        # annotations
        objects = util.parse_dota_poly2(file)
        for obj in objects:
            if int(obj['difficult']) == difficult:
                print('difficult: ', difficult)
                continue
            single_obj = {}
            # single_obj['area'] = obj['area']
            single_obj['category_id'] = cls_names.index(obj['name']) + 1
            single_obj['segmentation'] = []
            single_obj['segmentation'].append(obj['poly'])
            single_obj['iscrowd'] = 0
            xmin, ymin, xmax, ymax = min(obj['poly'][0::2]), min(obj['poly'][1::2]), \
                                     max(obj['poly'][0::2]), max(obj['poly'][1::2])

            width, height = xmax - xmin, ymax - ymin
            single_obj['bbox'] = xmin, ymin, width, height
            # modified
            single_obj['area'] = width * height
            single_obj['image_id'] = image_id
            data_dict['annotations'].append(single_obj)

        data_queue.put(data_dict)
    data_queue.put(None)


def anno_process_collect(meta, data_queue, destfile, process_n):
    inst_count = 1
    n = 0
    # data_dict = dict(images=[], categories=[], annotations=[])
    while True:
        data = data_queue.get()
        if data:
            meta['images'].append(data['images'])
            for anno in data['annotations']:
                anno['id'] = inst_count
                inst_count = inst_count + 1
                meta['annotations'].append(anno)
        else:
            n += 1
        if n == process_n:
            break
    with open(destfile, 'w') as f:
        json.dump(meta, f)


def img_process_unit(input_files, files_ids, data_queue):

    for file, image_id in zip(input_files, files_ids):
        data_dict = dict(annotations=[])

        img = cv2.imread(file)
        height, width, c = img.shape

        single_image = {}
        single_image['file_name'] = os.path.basename(file)
        single_image['id'] = image_id
        single_image['width'] = width
        single_image['height'] = height
        data_dict['images'] = single_image

        data_queue.put(data_dict)
    data_queue.put(None)


def img_process_collect(meta, data_queue, destfile, process_n):
    n = 0
    # data_dict = dict(images=[], categories=[], annotations=[])
    while True:
        data = data_queue.get()
        if data:
            meta['images'].append(data['images'])
        else:
            n += 1
        if n == process_n:
            break
    with open(destfile, 'w') as f:
        json.dump(meta, f)


def DOTA2COCO_ANNO_MULTI(srcpath, destfile, cls_names, difficult=0, ext='.png', processor=32):
    # set difficult to filter 2, 1, or do not filter, set 0

    imageparent = os.path.join(srcpath, 'images')
    labelparent = os.path.join(srcpath, 'labelTxt')

    data_dict = {}
    data_dict['images'] = []
    data_dict['categories'] = []
    data_dict['annotations'] = []
    for idex, name in enumerate(cls_names):
        single_cat = {'id': idex + 1, 'name': name, 'supercategory': name}
        data_dict['categories'].append(single_cat)

    filenames = util.GetFileFromThisRootDir(labelparent)
    fileids = list(range(1, len(filenames) + 1))
    process_payload = math.ceil(len(filenames) / processor)
    filelist = [filenames[i * process_payload:(i + 1) * process_payload] for i in range(processor)]
    fileidlist = [fileids[i * process_payload:(i + 1) * process_payload] for i in range(processor)]
    processor_pool = []
    data_queue = Queue(-1)
    collector = Process(target=anno_process_collect, args=(data_dict, data_queue, destfile, processor))
    collector.start()
    for i in range(processor):
        pro = Process(target=anno_process_unit, args=(imageparent, filelist[i], fileidlist[i], cls_names, data_queue, difficult, ext))
        pro.start()
        processor_pool.append(pro)

    for p in processor_pool:
        p.join()
    collector.join()


def DOTA2COCO_ONLYIMG_MULTI(srcpath, destfile, cls_names, processor=32):
    imageparent = os.path.join(srcpath, 'images')

    data_dict = {}
    data_dict['images'] = []
    data_dict['categories'] = []
    for idex, name in enumerate(cls_names):
        single_cat = {'id': idex + 1, 'name': name, 'supercategory': name}
        data_dict['categories'].append(single_cat)

    filenames = util.GetFileFromThisRootDir(imageparent)
    fileids = list(range(1, len(filenames) + 1))
    process_payload = math.ceil(len(filenames) / processor)
    filelist = [filenames[i * process_payload:(i + 1) * process_payload] for i in range(processor)]
    fileidlist = [fileids[i * process_payload:(i + 1) * process_payload] for i in range(processor)]
    processor_pool = []
    data_queue = Queue(-1)
    collector = Process(target=img_process_collect, args=(data_dict, data_queue, destfile, processor))
    collector.start()
    for i in range(processor):
        pro = Process(target=img_process_unit, args=(filelist[i], fileidlist[i], data_queue))
        pro.start()
        processor_pool.append(pro)

    for p in processor_pool:
        p.join()
    collector.join()

if __name__ == '__main__':
    # DOTA2COCO_ANNO_MULTI(r'/data/dota10/dota10_1024_200_s/trainval1024', r'/data/dota10/dota10_1024_200_s/trainval1024/DOTA_trainval1024.json', wordname_15, difficult='2')
    DOTA2COCO_ONLYIMG_MULTI(r'/data/dota10/dota10_1024_200_s/test1024', r'/data/dota10/dota10_1024_200_s/test1024/DOTA_test1024.json', wordname_15)
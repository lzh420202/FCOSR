import pickle
import os
from pycocotools.coco import COCO
import cv2
import shutil
import numpy as np
from DOTA_devkit.dota_evaluation import voc_eval, coco_eval
from DOTA_devkit.ResultMerge import mergebypolywithnms
from DOTA_devkit.ResultMerge_multi_process import mergebypoly as mergebypoly_multi_process
import json
import mmcv
import math
from multiprocessing import Pool
from hrsc2016_evaluation import hrsc2016_evaluate

dota_10 = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship',
           'tennis-court', 'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor',
           'swimming-pool', 'helicopter']

dota_15 = dota_10 + ['container-crane']

dota_20 = dota_15 + ['airport', 'helipad']

hrsc2016 = ['ship']

color_map = [(62, 39, 169),
             (69, 55, 214),
             (72, 76, 241),
             (69, 99, 253),
             (51, 123, 254),
             (44, 145, 240),
             (33, 164, 228),
             (11, 180, 211),
             (20, 191, 185),
             (53, 200, 155),
             (94, 205, 117),
             (151, 203, 73),
             (203, 193, 40),
             (244, 186, 57),
             (254, 204, 51),
             (246, 229, 39)]

def load_result(file):
    with open(file, 'rb') as f:
        data = pickle.load(f)
    return data

def drawResult(anno_file, result_file, save_dir):
    data = load_result(result_file)
    coco = COCO(anno_file)
    src_img = os.path.join(os.path.dirname(anno_file), 'images')
    class_name = coco.dataset['categories']
    if not os.path.exists(save_dir):
        print('create folder {}'.format(save_dir))
        os.makedirs(save_dir)
    det_folder = os.path.join(save_dir, 'det')
    no_det_folder = os.path.join(save_dir, 'no_det')
    if not os.path.exists(det_folder):
        print('create folder det in {}'.format(save_dir))
        os.makedirs(det_folder)
    if not os.path.exists(no_det_folder):
        print('create folder no_det in {}'.format(save_dir))
        os.makedirs(no_det_folder)

    for img in coco.dataset['images']:
        id = img['id'] - 1
        img_name = img['file_name']
        dst_name = os.path.splitext(img_name)[0] + ".jpg"
        img_full_path = os.path.join(src_img, img_name)
        image = cv2.imread(img_full_path)
        if data[id][0].shape[1] == 9:
            draw_flag = False
            for idx, result in enumerate(data[id]):
                if result.shape[0] == 0:
                    continue
                else:
                    for i in range(result.shape[0]):
                        bbox = result[i, :-1].reshape(-1, 2).round().astype(np.int32)
                        confidence = float(result[i, -1])
                        color = color_map[idx]
                        image = cv2.polylines(image, [bbox], True, color, 2)
                        label = class_name[idx]['name']
                        text = "{}:{:.2f}".format(label, confidence)
                        center = tuple(bbox.mean(0).round().astype(np.int32).tolist())
                        draw_flag = True
                        # cv2.putText(image, text, center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
            if draw_flag:
                cv2.imwrite(os.path.join(det_folder, dst_name), image)
            else:
                shutil.copy(img_full_path, os.path.join(no_det_folder, img_name))
        else:
            shutil.copy(img_full_path, os.path.join(no_det_folder, img_name))

def prepare_data_str(anno_file, result_file, cache_folder=None, type='dota_15', keep_ext=False, save_image_set=False):
    if cache_folder is None:
        _folder = os.path.dirname(result_file)
        basename = os.path.basename(result_file)
        cache_folder = os.path.join(_folder, os.path.splitext(basename)[0])
        if not os.path.exists(cache_folder):
            os.makedirs(cache_folder)

    coco = COCO(anno_file)
    data = load_result(result_file)
    if type == 'dota_15':
        class_list = dota_15
        result = dict(zip(dota_15, ['' for _ in range(16)]))
    elif type == 'dota_10':
        class_list = dota_10
        result = dict(zip(dota_10, ['' for _ in range(15)]))
    elif type == 'dota_20':
        class_list = dota_20
        raise NotImplementedError('dota 2.0.')
    elif type == 'hrsc2016':
        class_list = hrsc2016
    else:
        raise ValueError
    for i, d in enumerate(data):
        image_id = i + 1
        if keep_ext:
            image_name = coco.dataset['images'][i]['file_name']
        else:
            image_name = os.path.splitext(coco.dataset['images'][i]['file_name'])[0]
        if d[0].shape[1] == 9:
            for j, _d in enumerate(d):
                cat_id = j + 1
                cat_name = class_list[j]
                if _d.shape[0] == 0:
                    continue
                else:
                    for line_ in _d.tolist():
                        line = f'{image_name} {line_[8]} {line_[0]} {line_[1]} {line_[2]} {line_[3]} {line_[4]} {line_[5]} {line_[6]} {line_[7]}\n'
                        result[cat_name] += line

    for key in result.keys():
        R = result[key]
        with open(os.path.join(cache_folder, f'Task1_{key}.txt'), 'w') as f:
            f.write(R)

    if save_image_set:
        image_set = ''
        for key in coco.imgs.keys():
            if keep_ext:
                image_name = coco.imgs[key]['file_name']
            else:
                image_name = os.path.splitext(coco.imgs[key]['file_name'])[0]
            image_set += image_name + '\n'
        with open(os.path.join(os.path.dirname(cache_folder), 'image_set.txt'), 'w') as f:
            f.write(image_set)

    return cache_folder, os.path.join(cache_folder, 'image_set.txt')


def prepare_data_str_unit(split_data, coco, class_list, result, offset, keep_ext=False):
    for i, d in enumerate(split_data):
        image_id = i + 1
        if keep_ext:
            image_name = coco.dataset['images'][i + offset]['file_name']
        else:
            image_name = os.path.splitext(coco.dataset['images'][i + offset]['file_name'])[0]
        if d[0].shape[1] == 9:
            for j, _d in enumerate(d):
                cat_id = j + 1
                cat_name = class_list[j]
                if _d.shape[0] == 0:
                    continue
                else:
                    for line_ in _d.tolist():
                        line = f'{image_name} {line_[8]} {line_[0]} {line_[1]} {line_[2]} {line_[3]} {line_[4]} {line_[5]} {line_[6]} {line_[7]}\n'
                        result[cat_name] += line
    return result

def prepare_data_str_multi_process(anno_file, result_file, cache_folder=None, type='dota_15', keep_ext=False, save_image_set=False, process_num=16):
    if cache_folder is None:
        _folder = os.path.dirname(result_file)
        basename = os.path.basename(result_file)
        cache_folder = os.path.join(_folder, os.path.splitext(basename)[0])
        if not os.path.exists(cache_folder):
            os.makedirs(cache_folder)

    coco = COCO(anno_file)
    data = load_result(result_file)
    if type == 'dota_15':
        class_list = dota_15
    elif type == 'dota_10':
        class_list = dota_10
    elif type == 'dota_20':
        class_list = dota_20
        raise NotImplementedError('dota 2.0.')
    elif type == 'hrsc2016':
        class_list = hrsc2016
    else:
        raise ValueError
    result_split = [dict(zip(class_list, ['' for _ in range(len(class_list))])) for _1 in range(process_num)]
    n = math.ceil(len(data) / process_num)
    data_split_index = [pn * n for pn in list(range(process_num))] + [len(data)]
    data_split = [data[data_split_index[i]: data_split_index[i+1]] for i in range(process_num)]
    pool = Pool(process_num)
    all_result = []
    for i in range(process_num):
        result_ = pool.apply_async(prepare_data_str_unit, args=(data_split[i], coco, class_list, result_split[i], data_split_index[i], False, ))
        all_result.append(result_)
    pool.close()
    pool.join()
    result = dict(zip(class_list, ['' for _ in range(len(class_list))]))
    for result__ in all_result:
        result_decode = result__.get()
        for key in result_decode.keys():
            result[key] += result_decode[key]

    for key in result.keys():
        R = result[key]
        with open(os.path.join(cache_folder, f'Task1_{key}.txt'), 'w') as f:
            f.write(R)

    if save_image_set:
        image_set = ''
        for key in coco.imgs.keys():
            if keep_ext:
                image_name = coco.imgs[key]['file_name']
            else:
                image_name = os.path.splitext(coco.imgs[key]['file_name'])[0]
            image_set += image_name + '\n'
        with open(os.path.join(os.path.dirname(cache_folder), 'image_set.txt'), 'w') as f:
            f.write(image_set)

    return cache_folder, os.path.join(cache_folder, 'image_set.txt')

def merge_result(coco_anno_file, result_file, type, nms_thresh=0.3, remove_cache=True):
    assert os.path.isabs(result_file)
    result_folder = os.path.splitext(result_file)[0]
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    cache_folder = os.path.join(result_folder, 'cache')
    if not os.path.exists(cache_folder):
        os.makedirs(cache_folder)
    prepare_data_str(coco_anno_file, result_file, cache_folder, type, False, False)
    mergebypolywithnms(cache_folder, result_folder, nms_thresh)
    if remove_cache:
        shutil.rmtree(cache_folder)

def merge_result_multi_process(coco_anno_file, result_file, type, nms_thresh=0.3, remove_cache=True, process_num=36):
    assert os.path.isabs(result_file)
    result_folder = os.path.splitext(result_file)[0]
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    cache_folder = os.path.join(result_folder, 'cache')
    if not os.path.exists(cache_folder):
        os.makedirs(cache_folder)
    prepare_data_str_multi_process(coco_anno_file, result_file, cache_folder, type, False, False, process_num)
    mergebypoly_multi_process(cache_folder, result_folder)
    if remove_cache:
        shutil.rmtree(cache_folder)

def evaluateResult(coco_anno_file, result_file, anno_folder, type='dota_15'):
    file_folder, imagesetfile = prepare_data_str(coco_anno_file, result_file, type=type, keep_ext=False, save_image_set=True)
    file_src = os.path.join(file_folder, 'task1_{}.txt')
    anno_src = os.path.join(anno_folder, '{}.txt')
    coco = COCO(coco_anno_file)
    class_name = coco.dataset['categories']
    map = 0.0
    classaps = []
    names = []
    for class__ in class_name:
        classname = class__['name']
        names.append(classname)
        print('classname:', classname)
        rec, prec, ap = voc_eval(file_src,
             anno_src,
             imagesetfile,
             classname,
             ovthresh=0.5,
             use_07_metric=True)
        map = map + ap
        #print('rec: ', rec, 'prec: ', prec, 'ap: ', ap)
        print('ap: ', ap)
        classaps.append(ap)
        # umcomment to show p-r curve of each category
        # plt.figure(figsize=(8,4))
        # plt.xlabel('recall')
        # plt.ylabel('precision')
        # plt.plot(rec, prec)
       # plt.show()
    map = map/len(class_name)
    print('map:', map)
    classaps = 100*np.array(classaps)
    print('classaps: ', classaps)
    result_dir = os.path.dirname(result_file)
    basename = os.path.splitext(os.path.basename(result_file))[0]
    with open(os.path.join(result_dir, f'{basename}_evaluate.txt'), 'w') as f:
        result_json = dict(file=f'{basename}.pkl', mAp=map, detail=dict(zip(names, classaps)))
        json.dump(result_json, f)
    shutil.rmtree(file_folder)

def prepare_data(detection, imagenames, catnames, keep_ext=False):
    result = dict(zip(catnames, [{'name': [], 'confidence': [], 'detection': []} for _ in range(len(catnames))]))
    for i, d in enumerate(detection):
        if keep_ext:
            image_name = imagenames[i]
        else:
            image_name = os.path.splitext(imagenames[i])[0]
        if d[0].shape[1] == 9:
            for j, _d in enumerate(d):
                if j >= len(catnames):
                    continue
                cat_name = catnames[j]
                if _d.shape[0] == 0:
                    continue
                else:
                    for line_ in _d.tolist():
                        result[cat_name]['name'].append(image_name)
                        result[cat_name]['confidence'].append(line_[8])
                        result[cat_name]['detection'].append(line_[:-1])
    return result

def evaluateCOCO(coco_anno_file, result_file, iou_thrs=[0.5]):
    detection = load_result(result_file)
    coco = COCO(coco_anno_file)
    if iou_thrs is None:
        iou_thrs = [0.5]
    map = [0.0 for _ in iou_thrs]
    classaps = [[] for _ in iou_thrs]
    imagenames = [line['file_name'] for line in coco.dataset['images']]
    catnames = [line['name'] for line in coco.dataset['categories']]
    d = prepare_data(detection, imagenames, catnames)

    # class_name = coco.dataset['categories']
    # map = 0.0
    # classaps = []
    names = []
    for classname in catnames:
        names.append(classname)
        print('classname:', classname)
        for i, ovthresh in enumerate(iou_thrs):
            rec, prec, ap = coco_eval(d,
                                      coco,
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
    print('classaps: ', classaps)
    result_json = {}
    for i, ovthresh in enumerate(iou_thrs):
        result_json[f'iou_{ovthresh * 100:.0f}'] = dict(mAp=map[i], detail=dict(zip(names, classaps[i])))
    return result_json

def prase_config(file, root):
    cfg = mmcv.Config.fromfile(file)
    checkpoint = os.path.join(cfg['work_dir'], 'epoch_{}.pth')
    result = os.path.join(cfg['work_dir'], 'result_{}.pkl')
    draw_folder = os.path.join(cfg['work_dir'], 'visual_epoch_{}')
    if not os.path.isabs(checkpoint):
        checkpoint = os.path.join(root, checkpoint)
    if not os.path.isabs(result):
        result = os.path.join(root, result)
    if not os.path.isabs(draw_folder):
        draw_folder = os.path.join(root, draw_folder)
    return checkpoint, result, draw_folder

def prase_config_hrsc(file, root):
    cfg = mmcv.Config.fromfile(file)
    checkpoint = os.path.join(cfg['work_dir'], 'iter_{}.pth')
    result = os.path.join(cfg['work_dir'], 'result_{}.pkl')
    draw_folder = os.path.join(cfg['work_dir'], 'visual_iter_{}')
    if not os.path.isabs(checkpoint):
        checkpoint = os.path.join(root, checkpoint)
    if not os.path.isabs(result):
        result = os.path.join(root, result)
    if not os.path.isabs(draw_folder):
        draw_folder = os.path.join(root, draw_folder)
    return checkpoint, result, draw_folder

def main():
    mode = 'test'
    draw_result_sub = False
    draw_result_full = False
    data_src = r''
    data_type = 'dota_15'
    multi_scale = True
    data_root = dict(dota_10=dict(single=r'data/dota10_1024_4', multi_scale=r'data/dota10_1024_ms_2'),
                     dota_15=dict(single=r'data/dota15_1024', multi_scale=r'data/dota10_1024_ms_3'))
    merge_parallel = True
    process_num = 32

    root = '/workspace/mmdetection-2.15.1'
    config_folder = os.path.join(root, 'configs/fcosrbox')
    epochs = list(range(36, 35, -1))
    use_gpus = [0, 1, 2, 3]
    config_file_name = 'fcosr_mobilenetv2_fpn_8_128_3x_iou_rotate_new_data_drop_ps0.6_local_1.5_ms_v2.py'
    config_file = os.path.join(config_folder, config_file_name)

    if data_type == 'dota_10':
        name = '1_0'
        data_path = data_root['dota_10']
    elif data_type == 'dota_15':
        name = '1_5'
        data_path = data_root['dota_15']
    else:
        raise ValueError()
    if multi_scale:
        ms = '_ms'
        data_path = data_path['multi_scale']
    else:
        ms = ''
        data_path = data_path['single']
    if mode == 'test':
        anno_file = os.path.join(data_path, f"test1024{ms}/DOTA{name}_test1024{ms}.json")
    elif mode == 'val':
        anno_file = os.path.join(data_path, f"val1024{ms}/DOTA{name}_val1024{ms}.json")
    else:
        raise ValueError
    gpu_n = len(use_gpus)
    gpus = ','.join([str(n) for n in use_gpus])
    checkpoint_template, result_template, vis_template = prase_config(config_file, root)
    for epoch in epochs:
        print('*' * 10 + f'Start epoch: {epoch}' + '*' * 10)
        checkpoint_file = checkpoint_template.format(epoch)
        result_file = result_template.format(epoch)
        os.system(f'export CUDA_VISIBLE_DEVICES="{gpus}" && ./tools/dist_test.sh {config_file} {checkpoint_file} {gpu_n} --out {result_file}')
        print('*' * 10 + f'Epoch {epoch} finish!' + '*' * 10)
        print('*' * 10 + f'Epoch {epoch} evaluating!' + '*' * 10)
        if draw_result_sub:
            from multiprocessing import Process
            draw_p = Process(target=drawResult, args=(anno_file, result_file, vis_template.format(epoch)))
            draw_p.start()
        if mode == 'test':
            if merge_parallel:
                merge_result_multi_process(anno_file, result_file, data_type, 0.1, True, process_num)
            else:
                merge_result(anno_file, result_file, data_type, 0.1, True)
        elif mode == 'val':
            result_json = evaluateCOCO(anno_file, result_file, [0.5])
            with open(os.path.join(os.path.dirname(result_file), f'evaluate_{epoch}.json'), 'w') as f:
                json.dump(result_json, f)
            print('*' * 10 + f'Epoch {epoch} evaluate finished.' + '*' * 10)
        if draw_result_sub:
            draw_p.join()

def hrsc2016_evaluation_main():
    draw_result = False
    data_path = r'data/HRSC2016_COCO'

    root = '/workspace/mmdetection-2.15.1'
    config_folder = os.path.join(root, 'configs/fcosrbox')
    # epochs = list(range(36, 35, -1))
    iters = list(range(40000, 14000, -1000))
    use_gpus = [0, 1, 2, 3]
    config_file_name = 'fcosr_rx50_32x4d_fpn_40k_hrsc2016.py'
    config_file = os.path.join(config_folder, config_file_name)
    anno_file = os.path.join(data_path, f"test/HRSC_L1_test.json")

    gpu_n = len(use_gpus)
    gpus = ','.join([str(n) for n in use_gpus])
    checkpoint_template, result_template, vis_template = prase_config_hrsc(config_file, root)
    for iter_ in iters:
        print('*' * 10 + f'Start iter: {iter_}' + '*' * 10)
        checkpoint_file = checkpoint_template.format(iter_)
        result_file = result_template.format(iter_)
        os.system(f'export CUDA_VISIBLE_DEVICES="{gpus}" && ./tools/dist_test.sh {config_file} {checkpoint_file} {gpu_n} --out {result_file}')
        print('*' * 10 + f'Iter {iter_} finish!' + '*' * 10)
        print('*' * 10 + f'Iter {iter_} evaluating!' + '*' * 10)
        if draw_result:
            from multiprocessing import Process
            draw_p = Process(target=drawResult, args=(anno_file, result_file, vis_template.format(iter_)))
            draw_p.start()
        cache_folder, _ = prepare_data_str_multi_process(anno_file, result_file, type='hrsc2016', keep_ext=False)
        ap = hrsc2016_evaluate(cache_folder, os.path.join(data_path, 'test/labelTxt'))
        with open(os.path.join(cache_folder, f'evaluate_{iter_}.json'), 'w') as f:
            json.dump(ap, f)
        print('*' * 10 + f'Iter {iter_} evaluate finished.' + '*' * 10)
        if draw_result:
            draw_p.join()

if __name__ == '__main__':
    hrsc2016_evaluation_main()
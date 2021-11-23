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
from mmcv import Config

dota_10 = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship',
           'tennis-court', 'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor',
           'swimming-pool', 'helicopter']

dota_15 = dota_10 + ['container-crane']

dota_20 = dota_15 + ['airport', 'helipad']

color_map = [
    (54, 67, 244),
    (99, 30, 233),
    (176, 39, 156),
    (183, 58, 103),
    (181, 81, 63),
    (243, 150, 33),
    (212, 188, 0),
    (136, 150, 0),
    (80, 175, 76),
    (74, 195, 139),
    (57, 220, 205),
    (59, 235, 255),
    (0, 152, 255),
    (34, 87, 255),
    (72, 85, 121),
    (139, 125, 96),
    (246, 229, 39)]


def parse_args():
    parser = argparse.ArgumentParser(description='prepare dota')
    parser.add_argument('config', type=str, help='Path to configure file.')
    parser.add_argument('checkpoint', type=str, help='Check point file path.')
    parser.add_argument('-d', '--draw', action='store_true', default=False, help='Draw flag.', dest='draw_flag')
    parser.add_argument('-n', '--draw-num', default='all', help='Draw number.', dest='draw_n')
    parser.add_argument('-s', '--source-dir', type=str, help='Source image dir.', dest='source_dir')
    parser.add_argument('-D', '--draw-dir', type=str, help='Draw result dir.', dest='draw_dir')
    parser.add_argument('-g', '--gpus', default='0', help='GPUs, use common to separate devices.')
    parser.add_argument('-p', '--processor-num', type=int, default=16, help='processor number', dest='processor')
    args = parser.parse_args()
    return args


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


def draw_unite(imagefile, result_list_dict, save_dir):
    filename = os.path.splitext(os.path.basename(imagefile))[0]
    if result_list_dict.get(filename):
        img = cv2.imread(imagefile)
        for result in result_list_dict[filename]:
            label = result[0]
            points = np.array(list(map(lambda x: float(x), result[2:]))).reshape(-1, 2).round().astype(np.int32)
            img = cv2.polylines(img, [points], True, color_map[dota_20.index(label)], 4)
        cv2.imwrite(os.path.join(save_dir, f"{filename}_result.jpg"), img)


def parse_all_result(result_dir):
    file_list = [file for file in os.listdir(result_dir) if file.lower().endswith('.txt')]
    result_dict = {}
    for file in file_list:
        with open(os.path.join(result_dir, file), 'r') as f:
            all_results = [line.strip().split(" ") for line in f.readlines() if len(line.strip()) > 0]
        labelname = os.path.splitext(file)[0].split('Task1_')[1]
        for result in all_results:
            if not result[0] in result_dict.keys():
                result_dict[result[0]] = [[labelname] + result[1:]]
            else:
                result_dict[result[0]].append([labelname] + result[1:])
    return result_dict


def drawWhole(result_dir, src_image_dir, save_dir, process_num=16, numbers='all'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    results = parse_all_result(result_dir)
    pool = Pool(process_num)
    if numbers == 'all':
        image_list = [os.path.join(src_image_dir, file) for file in os.listdir(src_image_dir) if file.lower().endswith(('.jpg', 'png', 'bmp'))]
    elif isinstance(numbers, int):
        image_list = [os.path.join(src_image_dir, file) for file in os.listdir(src_image_dir) if file.lower().endswith(('.jpg', 'png', 'bmp'))]
        image_list = image_list[:numbers]
    else:
        raise ValueError("numbers format error")
    for image_file in image_list:
        pool.apply_async(draw_unite, args=(image_file, results, save_dir))
    pool.close()
    pool.join()


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
    return result_folder


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
    return result_folder


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


def print_args(args):
    print_n = 30
    print(f"{''.join(['-'] * print_n)}")
    print(f"Config file: {args.config}")
    print(f"Checkpoint file: {args.checkpoint}")
    if args.draw_flag and args.source_dir:
        print("Draw result: True")
        print(f"Draw source folder: {args.source_dir}")
        print(f"Draw number: {args.draw_n}")
        if args.draw_dir is None:
            print(f"Draw folder: {os.path.join(os.path.splitext(args.checkpoint)[0], 'visual_result')}")
        else:
            print(f"Draw folder: {args.draw_dir}")
    print(f"GPUs: {args.gpus}")
    print(f"Processor number: {args.processor}")
    print(f"{''.join(['-'] * print_n)}")


def main(args):
    draw_result_full = args.draw_flag
    draw_num = args.draw_n
    src_image_dir = args.source_dir
    process_num = args.processor
    use_gpus = [int(gpu.strip()) for gpu in args.gpus.split(',') if len(gpu.strip()) > 0]
    draw_dir = args.draw_dir
    checkpoint = args.checkpoint
    if draw_dir is None:
        draw_dir = os.path.join(os.path.splitext(checkpoint)[0], 'visual_result')

    config_file = args.config
    config = Config.fromfile(config_file)
    if config.dataset_type == 'DOTADataset_10':
        data_type = 'dota_10'
    elif config.dataset_type == 'DOTADataset_15':
        data_type = 'dota_15'
    else:
        raise ValueError(f"Unsupport data type: {config.dataset_type}")

    anno_file = config.data.test.ann_file
    gpu_n = len(use_gpus)
    gpus = ','.join([str(n) for n in use_gpus])

    print('*' * 10 + 'Start inference' + '*' * 10)
    result_file = os.path.splitext(checkpoint)[0] + '.pkl'
    os.system(f'export CUDA_VISIBLE_DEVICES="{gpus}" && ./tools/dist_test.sh {config_file} {checkpoint} {gpu_n} --out {result_file} --fuse-conv-bn')
    print('*' * 10 + 'Done!' + '*' * 10)
    print('*' * 10 + 'Evaluating!' + '*' * 10)

    result_dir = merge_result_multi_process(anno_file, result_file, data_type, 0.1, True, process_num)
    if draw_result_full and src_image_dir:
        if not os.path.exists(draw_dir):
            os.makedirs(draw_dir)
        assert draw_num == 'all' or isinstance(draw_num, int)
        print(f'Drawing result on the whole image. Number: {draw_num}')
        drawWhole(result_dir, src_image_dir, draw_dir, 16, draw_num)
        print('All done!')

if __name__ == '__main__':
    args = parse_args()
    print_args(args)
    main(args)
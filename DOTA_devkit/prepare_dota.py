import utils as util
import os
from ImgSplit_multi_process import splitbase
from SplitOnlyImage_multi_process import splitbase as split_img
import shutil
from multiprocessing import Pool
from DOTA2COCO import (DOTA2COCO_ANNO, DOTA2COCO_ONLYIMG, wordname_15, wordname_16, wordname_18)
from DOTA2COCO_multi_process import (DOTA2COCO_ANNO_MULTI, DOTA2COCO_ONLYIMG_MULTI)
import argparse
USE_MULTI_PROCESS = False


def parse_args():
    parser = argparse.ArgumentParser(description='prepare dota')
    parser.add_argument('srcpath', help='Source path.')
    parser.add_argument('dstpath', help='DOTA save path.')
    parser.add_argument('-s', '--subsize', type=int, default=1024, help='patch size')
    parser.add_argument('-g', '--gap', type=int, default=200, help='gap')
    parser.add_argument('-t', '--type', default='dota10', help='data type')
    parser.add_argument('-p', '--processor-num', type=int, default=16, help='processor number', dest='processor')
    parser.add_argument('-d', '--drop-threshold', type=float, default=0.5, help='drop threshold', dest='drop_thre')
    parser.add_argument('--scales', nargs='+', type=float, help='multi-scales')
    args = parser.parse_args()
    return args


def print_arg(args):
    print_n = 30
    print(f"{''.join(['-']*print_n)}")
    print(f"Source path: {args.srcpath}")
    print(f"Destination path: {args.dstpath}")
    print(f"Sub patch size: {args.subsize}")
    print(f"Gaps: {args.gap}")
    print(f"Data type: {args.type}")
    print(f"Processor number: {args.processor}")
    if args.scales:
        print(f"Multi scales: {', '.join([str(scale) for scale in args.scales])}")
    else:
        print("Multi scale: False")
    print(f"{''.join(['-'] * print_n)}")


def single_copy(src_dst_tuple):
    shutil.copyfile(*src_dst_tuple)


def filecopy(srcpath, dstpath, num_process=32):
    pool = Pool(num_process)
    filelist = util.GetFileFromThisRootDir(srcpath)

    name_pairs = []
    for file in filelist:
        basename = os.path.basename(file.strip())
        dstname = os.path.join(dstpath, basename)
        name_tuple = (file, dstname)
        name_pairs.append(name_tuple)

    pool.map(single_copy, name_pairs)


def singel_move(src_dst_tuple):
    shutil.move(*src_dst_tuple)


def filemove(srcpath, dstpath, num_process=32):
    pool = Pool(num_process)
    filelist = util.GetFileFromThisRootDir(srcpath)

    name_pairs = []
    for file in filelist:
        basename = os.path.basename(file.strip())
        dstname = os.path.join(dstpath, basename)
        name_tuple = (file, dstname)
        name_pairs.append(name_tuple)

    pool.map(filemove, name_pairs)


def getnamelist(srcpath, dstfile):
    filelist = util.GetFileFromThisRootDir(srcpath)
    with open(dstfile, 'w') as f_out:
        for file in filelist:
            basename = util.mybasename(file)
            f_out.write(basename + '\n')


def prepare_single_scale(srcpath, dstpath, subsize=1024, gap=200, data_type='dota10', drop_thre=0.5, num_process=16):
    """
    :param srcpath: train, val, test
          train --> trainval1024, val --> trainval1024, test --> test1024
    :return:
    """
    assert data_type in ['dota10', 'dota15', 'dota20']

    dst_train_folder = os.path.join(dstpath, f'trainval{subsize}')
    dst_test_folder = os.path.join(dstpath, f'test{subsize}')
    if not os.path.exists(dst_train_folder):
        os.makedirs(dst_train_folder)
    if not os.path.exists(dst_test_folder):
        os.makedirs(dst_test_folder)
    if not os.path.exists(os.path.join(dst_test_folder, 'images')):
        os.makedirs(os.path.join(dst_test_folder, 'images'))
    src_train_folder = os.path.join(srcpath, 'train')
    src_val_folder = os.path.join(srcpath, 'val')
    src_test_folder = os.path.join(srcpath, 'test')

    split_train = splitbase(src_train_folder, dst_train_folder, gap=gap,
                            subsize=subsize, num_process=num_process, drop_thresh=drop_thre)
    split_train.splitdata(1.0)

    split_val = splitbase(src_val_folder, dst_train_folder, gap=gap,
                          subsize=subsize, num_process=num_process, drop_thresh=drop_thre)
    split_val.splitdata(1.0)

    split_test = split_img(os.path.join(src_test_folder, 'images'),
                           os.path.join(dst_test_folder, 'images'),
                           gap=gap, subsize=subsize, num_process=num_process)
    split_test.splitdata(1.0)
    if data_type == 'dota10':
        class_name = wordname_15
        filename = '1_0'
    elif data_type == 'dota15':
        class_name = wordname_16
        filename = '1_5'
    elif data_type == 'dota20':
        class_name = wordname_18
        filename = '2_0'
    else:
        raise ValueError
    print('Processing train data.')
    if USE_MULTI_PROCESS:
        DOTA2COCO_ANNO_MULTI(dst_train_folder, os.path.join(dst_train_folder, f'DOTA{filename}_trainval{subsize}.json'), class_name, difficult='2', processor=num_process)
    else:
        DOTA2COCO_ANNO(dst_train_folder, os.path.join(dst_train_folder, f'DOTA{filename}_trainval{subsize}.json'), class_name, difficult='2')
    print('Processing test data.')
    if USE_MULTI_PROCESS:
        DOTA2COCO_ONLYIMG_MULTI(dst_test_folder, os.path.join(dst_test_folder, f'DOTA{filename}_test{subsize}.json'), class_name)
    else:
        DOTA2COCO_ONLYIMG(dst_test_folder, os.path.join(dst_test_folder, f'DOTA{filename}_test{subsize}.json'), class_name)
    print('All done.')


def prepare_multi_scale(srcpath, dstpath, subsize=1024, gap=200, data_type='dota10', drop_thre=0.5, num_process=16, multi_scale=(0.5, 1.0)):
    """
    :param srcpath: train, val, test
          train --> trainval1024, val --> trainval1024, test --> test1024
    :return:
    """

    assert data_type in ['dota10', 'dota15', 'dota20']
    print(f'Type: {data_type}')

    dst_train_folder = os.path.join(dstpath, f'trainval{subsize}_ms')
    dst_test_folder = os.path.join(dstpath, f'test{subsize}_ms')
    if not os.path.exists(dst_train_folder):
        os.makedirs(dst_train_folder)
    if not os.path.exists(dst_test_folder):
        os.makedirs(dst_test_folder)
    if not os.path.exists(os.path.join(dst_test_folder, 'images')):
        os.makedirs(os.path.join(dst_test_folder, 'images'))
    src_train_folder = os.path.join(srcpath, 'train')
    src_val_folder = os.path.join(srcpath, 'val')
    src_test_folder = os.path.join(srcpath, 'test')

    split_train = splitbase(src_train_folder, dst_train_folder, gap=gap,
                            subsize=subsize, num_process=num_process, drop_thresh=drop_thre)
    print('Splitting train set.')
    for scale in multi_scale:
        split_train.splitdata(scale)

    split_val = splitbase(src_val_folder, dst_train_folder, gap=gap,
                          subsize=subsize, num_process=num_process, drop_thresh=drop_thre)
    for scale in multi_scale:
        split_val.splitdata(scale)

    print('Splitting test set.')
    split_test = split_img(os.path.join(src_test_folder, 'images'),
                           os.path.join(dst_test_folder, 'images'),
                           gap=gap, subsize=subsize, num_process=num_process)
    for scale in multi_scale:
        split_test.splitdata(scale)

    if data_type == 'dota10':
        class_name = wordname_15
        filename = '1_0'
    elif data_type == 'dota15':
        class_name = wordname_16
        filename = '1_5'
    elif data_type == 'dota20':
        class_name = wordname_18
        filename = '2_0'
    else:
        raise ValueError
    print('Processing train data.')
    if USE_MULTI_PROCESS:
        DOTA2COCO_ANNO_MULTI(dst_train_folder, os.path.join(dst_train_folder, f'DOTA{filename}_trainval{subsize}_ms.json'),
                             class_name, difficult='2', processor=num_process)
    else:
        DOTA2COCO_ANNO(dst_train_folder, os.path.join(dst_train_folder, f'DOTA{filename}_trainval{subsize}_ms.json'), class_name, difficult='2')
    print('Processing test data.')
    if USE_MULTI_PROCESS:
        DOTA2COCO_ONLYIMG_MULTI(dst_test_folder, os.path.join(dst_test_folder, f'DOTA{filename}_test{subsize}_ms.json'), class_name)
    else:
        DOTA2COCO_ONLYIMG(dst_test_folder, os.path.join(dst_test_folder, f'DOTA{filename}_test{subsize}_ms.json'), class_name)
    print('All done.')


if __name__ == '__main__':
    args = parse_args()
    print_arg(args)
    srcpath = args.srcpath
    dstpath = args.dstpath
    subsize = args.subsize
    gap = args.gap
    num_process = args.processor
    drop_thre = args.drop_thre
    type = args.type
    if USE_MULTI_PROCESS:
        print('Multi process mode: enable.')
    if args.scales:
        prepare_multi_scale(srcpath, dstpath, subsize, gap, type, drop_thre, num_process, args.scales)
    else:
        prepare_single_scale(srcpath, dstpath, subsize, gap, type, drop_thre, num_process)
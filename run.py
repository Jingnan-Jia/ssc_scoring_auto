# -*- coding: utf-8 -*-
# @Time    : 3/3/21 12:25 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
import csv
import datetime
import glob
import itertools
import os
import shutil
import threading
import time
from typing import (List, Tuple, Union)

import SimpleITK as sitk
import monai
import numpy as np
import nvidia_smi
import pandas as pd
import torch
import torch.nn as nn
# import streamlit as st
import torchvision.models as models
from filelock import FileLock
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms import Resize, RandomRotation, RandomHorizontalFlip, RandomVerticalFlip, functional,CenterCrop
from varname import nameof
import confusion
from set_args import args

log_dict = {}  # a global dict to store variables saved to log files


def get_net(name, nb_cls):
    if 'vgg' in name:
        if name == 'vgg11_bn':
            if args.pretrained:
                net = models.vgg11_bn(pretrained=True, progress=True)

                # net.classifier[3] = torch.nn.Linear(in_features=4096, out_features=args.fc2_nodes),
                net.classifier[6] = torch.nn.Linear(in_features=args.fc2_nodes, out_features=3, bias=True)
            else:
                net = models.vgg11_bn(pretrained=args.pretrained, progress=True, num_classes=nb_cls)
        elif name == 'vgg16':
            if args.pretrained:
                net = models.vgg16(pretrained=True, progress=True)
                # net.classifier[3] = torch.nn.Linear(4096, args.fc2_nodes),
                net.classifier[6] = torch.nn.Linear(in_features=args.fc2_nodes, out_features=3, bias=True)
            else:
                net = models.vgg16(pretrained=args.pretrained, progress=True, num_classes=nb_cls)
        elif name == 'vgg19':
            if args.pretrained:
                net = models.vgg19(pretrained=True, progress=True)
                # net.classifier[3] = torch.nn.Linear(4096, args.fc2_nodes),
                net.classifier[6] = torch.nn.Linear(in_features=args.fc2_nodes, out_features=3, bias=True)
            else:
                net = models.vgg19(pretrained=args.pretrained, progress=True, num_classes=nb_cls)
        else:
            raise Exception("Wrong vgg net name specified ", name)
        net.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  # change in_features to 1
    elif name == 'resnet18':
        net = models.resnet18()
    elif name == 'alex':
        net = models.alexnet()
    elif name == 'squeezenet':
        net = models.squeezenet1_0()
    elif name == 'densenet161':
        net = models.densenet161()
    elif name == 'inception_v3':
        net = models.inception_v3()
    elif name == 'shufflenet_v2_x1_0':
        net = models.shufflenet_v2_x1_0()
    elif name == 'resnext50_32x4d':
        net = models.resnext50_32x4d(pretrained=args.pretrained, progress=True)
        net.fc = nn.Linear(512 * models.resnet.Bottleneck.expansion, nb_cls)
        net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    elif name == 'wide_resnet50_2':
        net = models.wide_resnet50_2()
    elif name == 'mnasnet1_0':
        net = models.mnasnet1_0()
    elif name == 'resnext101_32x8d':
        net = models.resnext101_32x8d(pretrained=args.pretrained, progress=True)
        net.fc = nn.Linear(512 * models.resnet.Bottleneck.expansion, nb_cls)
        net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    else:
        raise Exception('Net name is not correct')

    return net


def load_itk(filename, require_sp_po=False):
    #     print('start load data')
    # Reads the image using SimpleITK
    if os.path.isfile(filename):
        itkimage = sitk.ReadImage(filename)

    else:
        print('nonfound:', filename)
        return [], [], []

    # Convert the image to a  numpy array first ands then shuffle the dimensions to get axis in the order z,y,x
    ct_scan = sitk.GetArrayFromImage(itkimage)

    # ct_scan[ct_scan>4] = 0 #filter trachea (label 5)
    # Read the origin of the ct_scan, will be used to convert the coordinates from world to voxel and vice versa.
    origin = np.array(list(reversed(itkimage.GetOrigin())))

    # Read the spacing along each dimension
    spacing = np.array(list(reversed(itkimage.GetSpacing())))
    #     print('get_orientation', get_orientation)
    if require_sp_po:
        return ct_scan, origin, spacing
    else:
        return ct_scan


def save_itk(filename, scan, origin, spacing, dtype='int16'):
    stk = sitk.GetImageFromArray(scan.astype(dtype))
    stk.SetOrigin(origin[::-1])
    stk.SetSpacing(spacing[::-1])

    writer = sitk.ImageFileWriter()
    writer.Execute(stk, filename, True)


def load_level_data_old(data_dir: str, label_file: str, level: int) -> Tuple[List, List]:
    """
    Load the data for the specific level.
    :param label_file:
    :param data_dir:
    :param level:
    :return: 
    """
    file_prefix = "Level" + str(level)
    # 3 neighboring slices for one level
    x_up = sorted(glob.glob(os.path.join(data_dir, "*", file_prefix + "_up*")))
    x_middle = sorted(glob.glob(os.path.join(data_dir, "*", file_prefix + "_middle*")))
    x_down = sorted(glob.glob(os.path.join(data_dir, "*", file_prefix + "_down*")))
    label_excel = pd.read_excel(label_file, engine='openpyxl')

    # 3 labels for one level
    y_disext = pd.DataFrame(label_excel, columns=['L' + str(level) + '_disext']).values
    y_gg = pd.DataFrame(label_excel, columns=['L' + str(level) + '_gg']).values
    y_retp = pd.DataFrame(label_excel, columns=['L' + str(level) + '_retp']).values

    y_disext = np.array(y_disext).reshape((-1,))
    y_gg = np.array(y_gg).reshape((-1,))
    y_retp = np.array(y_retp).reshape((-1,))

    x = sorted([*x_up, *x_middle, *x_down])

    # repeat each element of y 3 times
    y_disext = list(itertools.chain.from_iterable(itertools.repeat(x, 3) for x in y_disext))
    y_gg = list(itertools.chain.from_iterable(itertools.repeat(x, 3) for x in y_gg))
    y_retp = list(itertools.chain.from_iterable(itertools.repeat(x, 3) for x in y_retp))

    # y_disext = list(itertools.chain.from_iterable((itertools.repeat(y, 3) for y in y_disext)))
    # y_gg = list(itertools.chain.from_iterable((itertools.repeat(y, 3) for y in y_gg)))
    # y_retp = list(itertools.chain.from_iterable((itertools.repeat(y, 3) for y in y_retp)))

    y = [np.array([a, b, c]) for a, b, c in zip(y_disext, y_gg, y_retp)]

    assert os.path.dirname(x[0]) == os.path.dirname(x[1]) == os.path.dirname(x[2])
    assert len(x) == len(y)
    log_dict['patients_per_level'] = len(x)

    return x, y


def load_level_names_from_dir(data_dir: str, label_file: str, levels: Union[int, List[int]]) -> Tuple[List, List]:
    """
    Load the data for the specific level.
    :param label_file:
    :param data_dir:
    :param level:
    :return:
    """
    if type(levels) is list:
        x_list, y_list = [], []
        for level in levels:
            file_prefix = "Level" + str(level)
            x_middle = sorted(glob.glob(os.path.join(data_dir, "*", file_prefix + "_middle*")))
            label_excel = pd.read_excel(label_file, engine='openpyxl')

            # 3 labels for one level
            y_disext = pd.DataFrame(label_excel, columns=['L' + str(level) + '_disext']).values
            y_gg = pd.DataFrame(label_excel, columns=['L' + str(level) + '_gg']).values
            y_retp = pd.DataFrame(label_excel, columns=['L' + str(level) + '_retp']).values

            y_disext = np.array(y_disext).reshape((-1,))
            y_gg = np.array(y_gg).reshape((-1,))
            y_retp = np.array(y_retp).reshape((-1,))

            y = [np.array([a, b, c]) for a, b, c in zip(y_disext, y_gg, y_retp)]

            x_list.extend(x_middle)
            y_list.extend(y)

    assert len(x_list) == len(y_list)

    return x_list, y_list


def load_data_of_pats(dir_pats: List, label_file: str):
    df_excel = pd.read_excel(label_file, engine='openpyxl')
    df_excel = df_excel.set_index('PatID')
    x, y = [], []
    for dir_pat in dir_pats:
        # print(dir_pat)
        x_pat, y_pat = load_data_of_5_levels(dir_pat, df_excel)
        x.extend(x_pat)
        y.extend(y_pat)
    return x, y


def load_data_of_levels(level_names: List, y):
    level_middle = level_names
    level_all_names, level_all_y = [], []
    for level_m, y_ in zip(level_middle, y):
        level_idx = level_m.split('Level')[-1].split("_")[
            0]  # /data/jjia/ssc_scoring/dataset/SSc_DeepLearning/Pat_010/Level1_up.mha

        level_u = sorted(glob.glob(os.path.join(os.path.dirname(level_m), "Level" + level_idx + "_up*")))[0]
        # level_m = sorted(glob.glob(os.path.join(os.path.dirname(level_m), "Level" + level_idx + "_up*")))
        level_d = sorted(glob.glob(os.path.join(os.path.dirname(level_m), "Level" + level_idx + "_down*")))[0]

        level_all_names.extend([level_u, level_m, level_d])
        level_all_y.extend([y_, y_, y_])

    return level_all_names, level_all_y


def load_data_of_5_levels(dir_pat: str, df_excel: pd.DataFrame) -> Tuple[List, List]:
    x, y = [], []
    for level in [1, 2, 3, 4, 5]:
        x_level, y_level = load_data_of_a_level(dir_pat, df_excel, level)
        x.extend(x_level)
        y.extend(y_level)

    return x, y


def load_data_of_a_level_old(dir_pat: str, label_file: str, level: int) -> Tuple[List, List]:
    """
    Load the data for the specific level.
    :param dir_pat:
    :param label_file:
    :param level:
    :return:
    """
    file_prefix = "Level" + str(level)
    # 3 neighboring slices for one level
    x_up = sorted(glob.glob(os.path.join(dir_pat, file_prefix + "_up*")))
    x_middle = sorted(glob.glob(os.path.join(dir_pat, file_prefix + "_middle*")))
    x_down = sorted(glob.glob(os.path.join(dir_pat, file_prefix + "_down*")))
    label_excel = pd.read_excel(label_file, engine='openpyxl')

    # 3 labels for one level
    y_disext = pd.DataFrame(label_excel, columns=['L' + str(level) + '_disext']).values
    y_gg = pd.DataFrame(label_excel, columns=['L' + str(level) + '_gg']).values
    y_retp = pd.DataFrame(label_excel, columns=['L' + str(level) + '_retp']).values

    y_disext = np.array(y_disext).reshape((-1,))
    y_gg = np.array(y_gg).reshape((-1,))
    y_retp = np.array(y_retp).reshape((-1,))

    x = sorted([*x_up, *x_middle, *x_down])

    # repeat each element of y 3 times
    y_disext = list(itertools.chain.from_iterable(itertools.repeat(x, 3) for x in y_disext))
    y_gg = list(itertools.chain.from_iterable(itertools.repeat(x, 3) for x in y_gg))
    y_retp = list(itertools.chain.from_iterable(itertools.repeat(x, 3) for x in y_retp))

    y = [np.array([a, b, c]) for a, b, c in zip(y_disext, y_gg, y_retp)]

    assert os.path.dirname(x[0]) == os.path.dirname(x[1]) == os.path.dirname(x[2])
    assert len(x) == len(y)
    log_dict['patients_per_level'] = len(x)

    return x, y


def load_data_of_a_level(dir_pat: str, df_excel: pd.DataFrame, level: int) -> Tuple[List, List]:
    """
    Load the data for the specific level.
    :param df_excel:
    :param dir_pat:
    :param level:
    :return:
    """
    file_prefix = "Level" + str(level)
    # 3 neighboring slices for one level
    x_up = glob.glob(os.path.join(dir_pat, file_prefix + "_up*"))[0]
    x_middle = glob.glob(os.path.join(dir_pat, file_prefix + "_middle*"))[0]
    x_down = glob.glob(os.path.join(dir_pat, file_prefix + "_down*"))[0]
    x = [x_up, x_middle, x_down]

    excel = df_excel
    idx = int(dir_pat.split('/')[-1].split('Pat_')[-1])

    y_disext = excel.at[idx, 'L' + str(level) + '_disext']
    y_gg = excel.at[idx, 'L' + str(level) + '_gg']
    y_retp = excel.at[idx, 'L' + str(level) + '_retp']

    y_disext = [y_disext, y_disext, y_disext]
    y_gg = [y_gg, y_gg, y_gg]
    y_retp = [y_retp, y_retp, y_retp]

    y = [np.array([a, b, c]) for a, b, c in zip(y_disext, y_gg, y_retp)]

    assert os.path.dirname(x[0]) == os.path.dirname(x[1]) == os.path.dirname(x[2])
    assert len(x) == len(y)

    return x, y


def normalize(image):
    # normalize the image
    mean, std = np.mean(image), np.std(image)
    image = image - mean
    image = image / std
    return image


class SScScoreDataset(Dataset):
    """SSc scoring dataset."""

    def __init__(self, data_x_names, data_y_list, index: List = None, transform=None):

        self.data_x_names, self.data_y_list = np.array(data_x_names), np.array(data_y_list)
        lenth = len(self.data_x_names)
        if index is not None:
            self.data_x_names = self.data_x_names[index]
            self.data_y_list = self.data_y_list[index]
        print('loading data ...')
        self.data_x = [load_itk(x, require_sp_po=True) for x in self.data_x_names]
        self.data_x_np = [i[0] for i in self.data_x]
        self.data_x_or_sp = [[i[1], i[2]] for i in self.data_x]

        self.data_x_np = [normalize(x) for x in self.data_x_np]

        log_dict['normalize_data'] = True

        self.data_x_np = [x.astype(np.float32) for x in self.data_x_np]
        self.data_y_np = [y.astype(np.float32) for y in self.data_y_list]
        self.min = [np.min(x) for x in self.data_x_np]
        self.data_x_np = [np.pad(x, pad_width=((128, 128), (128, 128)), mode='constant', constant_values=(m,)) for x,m in zip(self.data_x_np, self.min)]
        self.data_x_tensor = [torch.as_tensor(x) for x in self.data_x_np]
        self.data_y_tensor = [torch.as_tensor(y) for y in self.data_y_np]

        # self.min_value = [torch.min(x).item() for x in self.data_x_tensor]  # min values after normalization
        # self.data_x_tensor = [functional.pad(x, padding=[128, 128], fill=min) for x, min in zip(self.data_x_tensor, self.min_value)]

        self.transform = transform

    def __len__(self):
        return len(self.data_y_tensor)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.data_x_tensor[idx]
        image_name = self.data_x_names[idx]
        image_origin, image_spacing = self.data_x_or_sp[idx]
        # image_info = {'name':image_name, 'origin':image_origin, 'spacing':image_spacing}
        label = self.data_y_tensor[idx]

        # if type(image_info['origin']) is torch.Tensor:
        #     image_info['origin'] = image_info['origin'].numpy()
        # if type(image_info['spacing']) is torch.Tensor:
        #     image_info['spacing'] = image_info['spacing'].numpy()

        # ori = list(image_info['origin'][0])
        image_origin = np.append(image_origin, 1)
        # sp = list(image_info['spacing'][0])
        image_spacing = np.append(image_spacing, 1)
        print(image_name)

        def crop_center(img, cropx, cropy):
            y, x = img.shape
            startx = x // 2 - (cropx // 2)
            starty = y // 2 - (cropy // 2)
            return img[starty:starty + cropy, startx:startx + cropx]
        img_before_aug = crop_center(image.numpy(), 512, 512)
        save_itk('aug_before_'+image_name.split('/')[-1],
                 img_before_aug, image_origin, image_spacing, dtype='float')


        if self.transform:
            image = self.transform(image)

        save_itk('aug_after_' + image_name.split('/')[-1],
                 image.numpy(), image_origin, image_spacing, dtype='float')

        # image = (image - torch.mean(image)) / torch.std(image)

        return image, label


class AddChannel:
    def __call__(self, img):
        """
        Apply the transform to `img`.
        """
        return img[None]


class Path():
    def __init__(self, id, check_id_dir=False):
        self.id = id
        self.slurmlog_dir = 'slurmlogs'
        self.model_dir = 'models'
        self.data_dir = 'dataset'

        self.id_dir = os.path.join(self.model_dir, str(int(id)), 'fold_' + str(args.fold))
        if args.mode == 'train' and check_id_dir:  # when infer, do not check
            if os.path.isdir(self.id_dir):  # the dir for this id already exist
                raise Exception('The same id_dir already exists', self.id_dir)

        for dir in [self.slurmlog_dir, self.model_dir, self.data_dir, self.id_dir]:
            if not os.path.isdir(dir):
                os.makedirs(dir)
                print('successfully create directory:', dir)

        self.model_fpath = os.path.join(self.id_dir, 'model.pt')

        self.train_batch_label = os.path.join(self.id_dir, 'train_batch_label.csv')
        self.train_batch_preds = os.path.join(self.id_dir, 'train_batch_preds.csv')
        self.train_batch_preds_int = os.path.join(self.id_dir, 'train_batch_preds_int.csv')
        self.train_batch_preds_end5 = os.path.join(self.id_dir, 'train_batch_preds_end5.csv')
        self.train_loss = os.path.join(self.id_dir, 'train_loss.csv')
        self.train_data = os.path.join(self.id_dir, 'train_data.csv')

        self.valid_batch_label = os.path.join(self.id_dir, 'valid_batch_label.csv')
        self.valid_batch_preds = os.path.join(self.id_dir, 'valid_batch_preds.csv')
        self.valid_batch_preds_int = os.path.join(self.id_dir, 'valid_batch_preds_int.csv')
        self.valid_batch_preds_end5 = os.path.join(self.id_dir, 'valid_batch_preds_end5.csv')
        self.valid_loss = os.path.join(self.id_dir, 'valid_loss.csv')
        self.valid_data = os.path.join(self.id_dir, 'valid_data.csv')

        self.test_batch_label = os.path.join(self.id_dir, 'test_batch_label.csv')
        self.test_batch_preds = os.path.join(self.id_dir, 'test_batch_preds.csv')
        self.test_batch_preds_int = os.path.join(self.id_dir, 'test_batch_preds_int.csv')
        self.test_batch_preds_end5 = os.path.join(self.id_dir, 'test_batch_preds_end5.csv')
        self.test_loss = os.path.join(self.id_dir, 'test_loss.csv')
        self.test_data = os.path.join(self.id_dir, 'test_data.csv')


def get_transform(mode=None):
    rotation = 90
    patch_size = 480
    image_size = 512
    vertflip = 0.5
    horiflip = 0.5
    xforms = [AddChannel()]
    if mode == 'train':
        xforms.extend([
            # RandomAffine(degrees=20, scale=(0.8, 1.2),
            # RandomCrop(patch_size),
            # CenterCrop(patch_size),
            # FiveCrop(patch_size), # may lead to more output
            RandomRotation(rotation),  # minimum value
            CenterCrop(image_size),
            RandomHorizontalFlip(p=horiflip),
            RandomVerticalFlip(p=vertflip),
            # ScaleIntensityRanged(keys[0], a_min=-1000.0, a_max=500.0, b_min=0.0, b_max=1.0, clip=True),
            # RandGaussianNoised(keys[0], prob=0.3, std=0.01),
            monai.transforms.RandGaussianNoise()
        ])

    transform = transforms.Compose(xforms)
    global log_dict
    log_dict['RandomVerticalFlip'] = vertflip
    log_dict['RandomHorizontalFlip'] = horiflip
    log_dict['RandomRotation'] = rotation
    log_dict['image_size'] = image_size
    # log_dict['RandomCrop'] = patch_size
    log_dict['RandGaussianNoise'] = 0.1

    return transform


def _bytes_to_megabytes(value_bytes):
    return round((value_bytes / 1024) / 1024, 2)


def record_GPU_info():
    if args.outfile:
        jobid_gpuid = args.outfile.split('-')[-1]
        tmp_split = jobid_gpuid.split('_')[-1]
        if len(tmp_split) == 2:
            gpuid = tmp_split[-1]
        else:
            gpuid = 0
        nvidia_smi.nvmlInit()
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(gpuid)
        gpuname = nvidia_smi.nvmlDeviceGetName(handle)
        gpuname = gpuname.decode("utf-8")
        log_dict['gpuname'] = gpuname
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        gpu_mem_usage = str(_bytes_to_megabytes(info.used)) + '/' + str(_bytes_to_megabytes(info.total)) + ' MB'
        log_dict['gpu_mem_usage'] = gpu_mem_usage
        gpu_util = 0
        for i in range(5):
            res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
            gpu_util += res.gpu
            time.sleep(1)
        gpu_util = gpu_util / 5
        log_dict['gpu_util'] = str(gpu_util) + '%'
    return None


def appendrows_to(fpath, data):
    with open(fpath, 'a') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        if len(data.shape) == 1:  # when data.shape==(batch_size,) in classification task
            data = data.reshape(-1, 1)
        writer.writerows(data)


def round_to_5(pred: Union[torch.Tensor, np.ndarray], device=torch.device("cpu")) -> Union[torch.Tensor, np.ndarray]:
    if type(pred) == torch.Tensor:
        tensor_flag = True
        pred = pred.cpu().detach().numpy()
    else:
        tensor_flag = False

    # elif type(pred) == np.ndarray:
    pred = np.rint(pred / 5) * 5
    pred[pred > 100] = 100
    pred[pred < 0] = 0

    if tensor_flag:
        pred = torch.tensor(pred)
        pred = pred.to(device)

    return pred


def get_dir_pats(data_dir: str, label_file: str) -> List:
    """
    get absolute directories of patients in this data_dir, use label_file to verify the existing directories.
    data_dir: relative path
    """
    abs_dir_path = os.path.dirname(os.path.realpath(__file__))  # abosolute path of the current .py file
    data_dir = abs_dir_path + "/" + data_dir
    dir_pats = sorted(glob.glob(os.path.join(data_dir, "Pat_*")))

    label_excel = pd.read_excel(label_file, engine='openpyxl')

    # 3 labels for one level
    pats_id_in_excel = pd.DataFrame(label_excel, columns=['PatID']).values
    pats_id_in_excel = [i[0] for i in pats_id_in_excel]
    assert len(dir_pats) == len(pats_id_in_excel)

    # assert the names of patients got from 2 ways
    pats_id_in_dir = [int(path.split('/')[-1].split('Pat_')[-1]) for path in dir_pats]
    pats_id_in_excel = [int(pat_id) for pat_id in pats_id_in_excel]
    assert pats_id_in_dir == pats_id_in_excel

    return dir_pats


def start_run(mode, net, dataloader, amp, epochs, device, loss_fun, loss_fun_mae, opt, scaler, mypath, epoch_idx,
              valid_mae_best=None):
    print(mode + "ing ......")
    if mode == 'train':
        net.train()
        loss_path = mypath.train_loss
    else:
        net.eval()
        if mode == 'valid':
            loss_path = mypath.valid_loss
        elif mode == 'test':
            loss_path = mypath.test_loss
        else:
            raise Exception('mode is wrong', mode)

    batch_idx = 0
    total_loss = 0
    total_loss_mae = 0
    total_loss_mae_end5 = 0
    for batch_x, batch_y in dataloader:


        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        if args.r_c == "c":
            batch_y = batch_y.type(torch.LongTensor)  # crossentropy requires LongTensor
            batch_y = batch_y.to(device)
        if amp:
            with torch.cuda.amp.autocast():
                if mode != 'train':
                    with torch.no_grad():
                        pred = net(batch_x)
                else:
                    pred = net(batch_x)

                loss = loss_fun(pred, batch_y)

                if args.r_c == "c":
                    pred = torch.argmax(pred, dim=1)
                    pred = pred.type(torch.FloatTensor)
                    pred = pred.to(device)
                    pred = pred * 5  # convert back to original scores
                    batch_y = batch_y * 5  # convert back to original scores
                    # pred = pred.type(torch.LongTensor)

                loss_mae = loss_fun_mae(pred, batch_y)
                pred_end5 = round_to_5(pred, device)
                loss_mae_end5 = loss_fun_mae(pred_end5, batch_y)
            if mode == 'train':  # update gradients only when training
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()

        else:
            if mode != 'train':
                with torch.no_grad():
                    pred = net(batch_x)
            else:
                pred = net(batch_x)

            loss = loss_fun(pred, batch_y)

            if args.r_c == "c":
                pred = torch.argmax(pred, dim=1)
                pred = pred.type(torch.FloatTensor)
                pred = pred.to(device)
                pred = pred * 5  # convert back to original scores
                batch_y = batch_y * 5  # convert back to original scores

            loss_mae = loss_fun_mae(pred, batch_y)
            pred_end5 = round_to_5(pred, device)
            loss_mae_end5 = loss_fun_mae(pred_end5, batch_y)

            if mode == 'train':  # update gradients only when training
                opt.zero_grad()
                loss.backward()
                opt.step()

        print(loss.item())

        total_loss += loss.item()
        total_loss_mae += loss_mae.item()
        total_loss_mae_end5 += loss_mae_end5.item()
        batch_idx += 1

        p1 = threading.Thread(target=record_GPU_info)
        p1.start()

    ave_loss = total_loss / batch_idx
    ave_loss_mae = total_loss_mae / batch_idx
    ave_loss_mae_end5 = total_loss_mae_end5 / batch_idx
    print("mode:", mode, "loss: ", ave_loss, "loss_mae: ", ave_loss_mae, "loss_mae_end5: ", ave_loss_mae_end5)

    if not os.path.isfile(loss_path):
        with open(loss_path, 'a') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(['step', 'loss', 'mae', 'mae_end5'])
    with open(loss_path, 'a') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow([epoch_idx, ave_loss, ave_loss_mae, ave_loss_mae_end5])

    if args.mode == 'train' and valid_mae_best is not None:
        if ave_loss_mae < valid_mae_best:
            print("old valid loss mae is: ", valid_mae_best)
            print("new valid loss mae is: ", ave_loss_mae)

            valid_mae_best = ave_loss_mae

            print('this model is the best one, save it. epoch id: ', epoch_idx)
            torch.save(net.state_dict(), mypath.model_fpath)
            print('save_successfully at ', mypath.model_fpath)
        return valid_mae_best
    else:
        return None


def get_column(n, tr_y):
    column = [i[n] for i in tr_y]
    column = [j / 5 for j in column]  # convert labels from [0,5,10, ..., 100] to [0, 1, 2, ..., 20]
    return column


def split_ts_data_by_levels(data_dir, label_file):
    level_x, level_y = load_level_names_from_dir(data_dir, label_file,
                                                 levels=[1, 2, 3, 4, 5])  # level names and level labels
    if args.ts_level_nb == 135:
        test_count = {0: 56, 5: 12, 10: 9, 15: 6, 20: 6, 25: 6, 30: 6, 35: 4, 40: 5, 45: 3, 50: 4, 55: 3, 60: 3, 65: 2,
                      70: 2, 75: 1, 80: 2, 85: 1, 90: 2, 95: 0, 100: 2}
    elif args.ts_level_nb == 235:
        test_count = {0: 115, 5: 24, 10: 18, 15: 9, 20: 12, 25: 9, 30: 9, 35: 6, 40: 8, 45: 3, 50: 7, 55: 3, 60: 3,
                      65: 2, 70: 2, 75: 1, 80: 1, 85: 1, 90: 2, 95: 0, 100: 1}

    tr_vd_x, tr_vd_y, test_x, test_y = [], [], [], []
    for x, y in zip(level_x, level_y):
        if test_count[y[0]] > 0:  # disext score > 0
            test_x.append(x)
            test_y.append(y)
            test_count[y[0]] -= 1
            continue
        else:
            tr_vd_x.append(x)
            tr_vd_y.append(y)
    tr_vd_x, tr_vd_y, test_x, test_y = map(np.array, [tr_vd_x, tr_vd_y, test_x, test_y])

    return tr_vd_x, tr_vd_y, test_x, test_y


def save_xy(tr_x, tr_y, mode, mypath):
    if mode == 'train':
        path = mypath.train_data
    elif mode == 'valid':
        path = mypath.valid_data
    elif mode == 'test':
        path = mypath.test_data
    else:
        raise Exception("wrong mode", mode)
    with open(path, 'a') as f:
        writer = csv.writer(f)
        for x, y in zip(tr_x, tr_y):
            writer.writerow([x, y])


def prepare_data():
    # get data_x names
    data_dir = "dataset/SSc_DeepLearning"
    label_file = "dataset/SSc_DeepLearning/GohScores.xlsx"
    log_dict['data_dir'] = data_dir
    log_dict['label_file'] = label_file
    # pat_names = get_dir_pats(data_dir, label_file)
    # pat_names = np.array(pat_names)

    tr_vd_level_names, tr_vd_level_y, test_level_names, test_level_y = split_ts_data_by_levels(data_dir, label_file)

    kf5 = KFold(n_splits=5, shuffle=True, random_state=42)  # for future reproduction
    log_dict['data_shuffle'] = True
    log_dict['data_shuffle_seed'] = 42
    kf_list = list(kf5.split(tr_vd_level_names))
    tr_level_idx, vd_level_idx = kf_list[args.fold - 1]

    log_dict['train_level_nb'] = len(tr_level_idx)
    log_dict['valid_level_nb'] = len(vd_level_idx)
    log_dict['test_level_nb'] = len(test_level_names)

    log_dict['train_index_head'] = tr_level_idx[:20]
    log_dict['valid_index_head'] = vd_level_idx[:20]

    tr_level_names = tr_vd_level_names[tr_level_idx]
    tr_level_y = tr_vd_level_y[tr_level_idx]
    vd_level_names = tr_vd_level_names[vd_level_idx]
    vd_level_y = tr_vd_level_y[vd_level_idx]

    tr_x, tr_y = load_data_of_levels(tr_level_names, tr_level_y)
    vd_x, vd_y = load_data_of_levels(vd_level_names, vd_level_y)
    ts_x, ts_y = load_data_of_levels(test_level_names, test_level_y)

    return tr_x, tr_y, vd_x, vd_y, ts_x, ts_y


def train(id):
    mypath = Path(id)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        amp = True
    else:
        device = "cpu"
        amp = False
    log_dict['amp'] = amp
    if args.r_c == "r":  # regression target will predict 3 labels at the same time.
        net = get_net(args.net, 3)
    else:
        net = get_net(args.net, 21)  # classification can only predict 21 classes of one label
    tr_x, tr_y, vd_x, vd_y, ts_x, ts_y = prepare_data()
    save_xy(tr_x, tr_y, 'train', mypath)
    save_xy(vd_x, vd_y, 'valid', mypath)
    save_xy(ts_x, ts_y, 'test', mypath)

    if args.r_c == "c":  # classification target
        if args.cls == "disext":
            tr_y, vd_y, ts_y = get_column(0, tr_y), get_column(0, vd_y), get_column(0, ts_y)
        elif args.cls == "gg":
            tr_y, vd_y, ts_y = get_column(1, tr_y), get_column(1, vd_y), get_column(1, ts_y)
        elif args.cls == "rept":
            tr_y, vd_y, ts_y = get_column(2, tr_y), get_column(2, vd_y), get_column(2, ts_y)
        else:
            raise Exception("Wrong args.cls: ", args.cls)

    if args.sampler:
        disext_list = []
        for sample in tr_y:
            if type(sample) in [list, np.ndarray]:
                disext_list.append(sample[0])
            else:
                disext_list.append(sample)
        disext_np = np.array(disext_list)
        disext_unique = np.unique(disext_np)
        class_sample_count = np.array([len(np.where(disext_np == t)[0]) for t in disext_unique])
        weight = 1. / class_sample_count
        disext_unique_list = list(disext_unique)
        samples_weight = np.array([weight[disext_unique_list.index(t)] for t in disext_np])

        # weight = [nb_nonzero/len(data_y_list) if e[0] == 0 else nb_zero/len(data_y_list) for e in data_y_list]
        samples_weight = samples_weight.astype(np.float32)
        samples_weight = torch.from_numpy(samples_weight)
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    else:
        sampler = None

    tr_dataset = SScScoreDataset(tr_x, tr_y, transform= get_transform('train'))
    vd_dataset = SScScoreDataset(vd_x, vd_y, transform= get_transform())
    ts_dataset = SScScoreDataset(ts_x, ts_y, transform= get_transform())

    batch_size = 10
    log_dict['batch_size'] = batch_size
    workers = 10
    log_dict['loader_workers'] = workers
    train_dataloader = DataLoader(tr_dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                                  sampler=sampler)
    valid_dataloader = DataLoader(vd_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    test_dataloader = DataLoader(ts_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)

    net = net.to(device)
    if args.r_c == "c":
        loss_fun = nn.CrossEntropyLoss()  # for classification task
        log_dict['loss_fun'] = 'CE'
    else:
        if args.loss == 'mae':
            loss_fun = nn.L1Loss()
        elif args.loss == 'smooth_mae':
            loss_fun = nn.SmoothL1Loss()
        elif args.loss == 'mse':
            loss_fun = nn.MSELoss()
        elif args.loss == 'mse+mae':
            loss_fun = nn.MSELoss() + nn.L1Loss()  # for regression task
        else:
            raise Exception("loss function is not correct ", args.loss)
    loss_fun_mae = nn.L1Loss()

    class MsePlusMae(nn.Module):
        """Dice and Xentropy loss"""

        def __init__(self):
            super().__init__()
            self.mse = nn.MSELoss()
            self.mae = nn.L1Loss()

        def forward(self, y_pred, y_true):
            mse = self.mse(y_pred, y_true)
            mae = self.mae(y_pred, y_true)
            print(f"mse loss: {mse}, mae loss: {mae}")
            return mse + mae

    lr = 1e-4
    log_dict[nameof(lr)] = lr
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    if amp:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    valid_mae_best = 10000
    if args.mode == 'train' or args.eval_id is None:
        epochs = args.epochs
    else:
        epochs = 1
    for i in range(epochs):  # 20000 epochs
        if args.mode != 'train':
            net.load_state_dict(torch.load(mypath.model_fpath, map_location=device))
        else:
            # run the training
            start_run('train', net, train_dataloader, amp, epochs, device, loss_fun, loss_fun_mae, opt, scaler, mypath,
                      i)
        # run the validation
        valid_mae_best = start_run('valid', net, valid_dataloader, amp, epochs, device, loss_fun, loss_fun_mae, opt,
                                   scaler, mypath, i, valid_mae_best)
        # run the testing
        # if i == epochs - 1:
        start_run('test', net, test_dataloader, amp, epochs, device, loss_fun, loss_fun_mae, opt, scaler, mypath, i)

    net.load_state_dict(torch.load(mypath.model_fpath, map_location=device))

    record_best_preds(net, train_dataloader, valid_dataloader, test_dataloader, mypath, device, amp)

    confusion.confusion(mypath.train_batch_label, mypath.train_batch_preds_end5)
    confusion.confusion(mypath.valid_batch_label, mypath.valid_batch_preds_end5)
    confusion.confusion(mypath.test_batch_label, mypath.test_batch_preds_end5)


def record_best_preds(net, train_dataloader, valid_dataloader, test_dataloader, mypath, device, amp):
    dataloader_dict = {'train': train_dataloader, 'valid': valid_dataloader, 'test': test_dataloader}

    for mode, dataloader in dataloader_dict.items():
        print("Start write pred to disk")
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            if args.r_c == "c":
                batch_y = batch_y.type(torch.LongTensor)  # crossentropy requires LongTensor
                batch_y = batch_y.to(device)
            if amp:
                with torch.cuda.amp.autocast():
                    with torch.no_grad():
                        pred = net(batch_x)
            else:
                with torch.no_grad():
                    pred = net(batch_x)
            if args.r_c == "c":
                pred = torch.argmax(pred, dim=1)
                pred = pred.type(torch.FloatTensor)
                pred = pred.to(device)
                pred = pred * 5  # convert back to original scores
                batch_y = batch_y * 5  # convert back to original scores

            record_preds(mode, batch_y, pred, mypath)

            if mode == 'train':
                p1 = threading.Thread(target=record_GPU_info)
                p1.start()


def record_preds(mode, batch_y, pred, mypath):
    batch_label = batch_y.cpu().detach().numpy().astype('Int64')
    batch_preds = pred.cpu().detach().numpy()
    batch_preds_int = batch_preds.astype('Int64')
    batch_preds_end5 = round_to_5(batch_preds_int)
    batch_preds_end5 = batch_preds_end5.astype('Int64')
    if mode == 'train':
        appendrows_to(mypath.train_batch_label, batch_label)
        appendrows_to(mypath.train_batch_preds, batch_preds)
        appendrows_to(mypath.train_batch_preds_int, batch_preds_int)
        appendrows_to(mypath.train_batch_preds_end5, batch_preds_end5)
    elif mode == 'valid':
        appendrows_to(mypath.valid_batch_label, batch_label)
        appendrows_to(mypath.valid_batch_preds, batch_preds)
        appendrows_to(mypath.valid_batch_preds_int, batch_preds_int)
        appendrows_to(mypath.valid_batch_preds_end5, batch_preds_end5)

    elif mode == 'test':
        appendrows_to(mypath.test_batch_label, batch_label)
        appendrows_to(mypath.test_batch_preds, batch_preds)
        appendrows_to(mypath.test_batch_preds_int, batch_preds_int)
        appendrows_to(mypath.test_batch_preds_end5, batch_preds_end5)


def record_experiment(record_file, id=None):
    if id is None:
        lock = FileLock(record_file + ".lock")
        with lock:  # with this lock,  open a file for exclusive access
            with open(record_file, 'a') as csv_file:
                if not os.path.isfile(record_file) or os.stat(record_file).st_size == 0:  # empty?
                    new_id = 1
                    df = pd.DataFrame()
                else:
                    df = pd.read_csv(record_file)
                    # df = df.fillna(-999)  # use -999 to replace empty element,
                    # cause empty cell would be read by np.nan, which makes the whole column become float forever
                    last_id = df['ID'].to_list()[-1]
                    new_id = int(last_id) + 1
                # df = df.astype(object)  # df need to be object, otherwise NAN cannot live in it
                mypath = Path(new_id, check_id_dir=True)  # to check if id_dir already exist

                date = datetime.date.today().strftime("%Y-%m-%d")
                time = datetime.datetime.now().time().strftime("%H:%M:%S")
                # row = [new_id, date, time, ]
                idatime = {'ID': new_id, 'start_date': date, 'start_time': time}

                args_dict = vars(args)
                idatime.update(args_dict)
                if len(df) == 0:
                    df = pd.DataFrame([idatime])  # need a [] , or need to assign the index for df
                else:
                    for key, value in idatime.items():
                        df.at[new_id - 1, key] = value  #
                    # df = df.append(idatime, ignore_index=True)  # would change the dtype of the whole column

                if args.mode == 'train':
                    for index, row in df.iterrows():
                        if 'State' not in list(row.index) or row['State'] in [None, np.nan, 'RUNNING']:
                            try:
                                jobid = row['outfile'].split('-')[-1].split('_')[0]  # extract job id from outfile name
                                seff = os.popen('seff ' + jobid)  # get job information
                                for line in seff.readlines():
                                    line = line.split(
                                        ': ')  # must have space to be differentiated from time format 00:12:34
                                    if len(line) == 2:
                                        key, value = line
                                        key = '_'.join(key.split(' '))  # change 'CPU utilized' to 'CPU_utilized'
                                        value = value.split('\n')[0]
                                        df.at[index, key] = value

                            except:
                                pass
                for column in df:
                    # try:
                    ori_type = type(df[column].to_list()[-1])
                    if ori_type is int:
                        df[column] = df[column].astype('Int64')  # correct type
                    # except:
                    #     pass
                # df = df.replace(-999, np.nan)
                df.to_csv(record_file, index=False)
                df.to_csv(mypath.id_dir + '/' + record_file, index=False)
                shutil.copy(record_file, 'cp_' + record_file)
        if new_id:
            return new_id
        else:
            raise Exception('ID is None')
    else:  # at the end of this experiments, find the line of this id, and record the final information
        lock = FileLock(record_file + ".lock")
        with lock:  # with this lock,  open a file for exclusive access
            df = pd.read_csv(record_file)
            # df = df.astype(object)
            # df = df.fillna(-999)  # use -999 to replace empty element,
            # cause empty cell would be read by np.nan, which makes the whole column become float forever
            index = df.index[df['ID'] == id].to_list()
            if len(index) > 1:
                raise Exception("over 1 row has the same id", id)
            elif len(index) == 0:  # only one line,
                index = 0
            else:
                index = index[0]

            date = datetime.date.today().strftime("%Y-%m-%d")
            time = datetime.datetime.now().time().strftime("%H:%M:%S")
            df.at[index, 'end_date'] = date
            df.at[index, 'end_time'] = time

            # usage
            f = "%Y-%m-%d %H:%M:%S"
            t1 = datetime.datetime.strptime(df['start_date'][index] + ' ' + df['start_time'][index], f)
            t2 = datetime.datetime.strptime(df['end_date'][index] + ' ' + df['end_time'][index], f)
            elapsed_time = check_time_difference(t1, t2)
            df.at[index, 'elapsed_time'] = elapsed_time

            mypath = Path(id)
            lock2 = FileLock(mypath.valid_loss + ".lock")
            with lock2:
                loss_df = pd.read_csv(mypath.valid_loss)
                best_index = loss_df['mae_end5'].idxmin()
                log_dict['valid_metrics_min'] = 'mae_end5'
                valid_loss = loss_df['loss'][best_index]
                valid_mae = loss_df['mae'][best_index]
                valid_mae_end5 = loss_df['mae_end5'][best_index]

            lock3 = FileLock(mypath.train_loss + ".lock")
            with lock3:
                loss_df = pd.read_csv(mypath.train_loss)
                best_index = loss_df['mae_end5'].idxmin()
                train_loss = loss_df['loss'][best_index]
                train_mae = loss_df['mae'][best_index]
                train_mae_end5 = loss_df['mae_end5'][best_index]

            lock4 = FileLock(mypath.test_loss + ".lock")
            with lock4:
                loss_df = pd.read_csv(mypath.test_loss)
                best_index = loss_df['mae_end5'].idxmin()
                test_loss = loss_df['loss'][best_index]
                test_mae = loss_df['mae'][best_index]
                test_mae_end5 = loss_df['mae_end5'][best_index]

            df.at[index, 'valid_loss'] = round(valid_loss, 2)
            df.at[index, 'valid_mae'] = round(valid_mae, 2)
            df.at[index, 'valid_mae_end5'] = round(valid_mae_end5, 2)

            df.at[index, 'train_loss'] = round(train_loss, 2)
            df.at[index, 'train_mae'] = round(train_mae, 2)
            df.at[index, 'train_mae_end5'] = round(train_mae_end5, 2)

            df.at[index, 'test_loss'] = round(test_loss, 2)
            df.at[index, 'test_mae'] = round(test_mae, 2)
            df.at[index, 'test_mae_end5'] = round(test_mae_end5, 2)

            # df = df.fillna(-999)  # use -999 to replace empty element,
            for key, value in log_dict.items():  # write all log_dict to csv file
                if type(value) is np.ndarray:
                    str_v = ''
                    for v in value:
                        str_v += str(v)
                        str_v += '_'
                    value = str_v
                df.loc[index, key] = value
                if type(value) is int:
                    df[key] = df[key].astype('Int64')

            for column in df:
                if type(df[column].to_list()[-1]) is int:
                    df[column] = df[column].astype('Int64')  # correct type

            args_dict = vars(args)
            args_dict.update({'ID': id})
            for column in df:
                if column in args_dict.keys() and type(args_dict[column]) is int:
                    df[column] = df[column].astype(float).astype('Int64')  # correct str to float and then int

            df.to_csv(record_file, index=False)
            df.to_csv(mypath.id_dir + '/' + record_file, index=False)
            shutil.copy(record_file, 'cp_' + record_file)
    # subprocess.run(["scp", record_file, "jjia@lkeb-std102:X:/research"])


def check_time_difference(t1: datetime, t2: datetime):
    t1_date = datetime.datetime(t1.year, t1.month, t1.day, t1.hour, t1.minute, t1.second)
    t2_date = datetime.datetime(t2.year, t2.month, t2.day, t2.hour, t2.minute, t2.second)
    t_elapsed = t2_date - t1_date

    return str(t_elapsed).split('.')[0]  # drop out microseconds


if __name__ == "__main__":
    if args.mode == 'train' or args.eval_id is None:
        record_file = 'records.csv'
        id = record_experiment(record_file)  # id is used to name moels/files/etc.
        train(id)
        record_experiment(record_file, id=id)
    else:
        record_file = 'record_infer.csv'
        record_experiment(record_file)
        id = args.eval_id
        train(id)

    print('finish this experiments! ')

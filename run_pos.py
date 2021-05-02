# -*- coding: utf-8 -*-
# @Time    : 3/3/21 12:25 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
import csv
import datetime
import glob
import os
import random
import shutil
import threading
import time
from typing import (Dict, List, Tuple, Hashable,
                    Optional, Sequence, Union, Mapping)

import SimpleITK as sitk
import numpy as np
import nvidia_smi
import pandas as pd
import torch
import torch.nn as nn
# import streamlit as st
from filelock import FileLock
from monai.transforms import ScaleIntensityRange, RandGaussianNoise
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import WeightedRandomSampler

import confusion
import jjnutils.util as futil
import pingouin as pg
from set_args_pos import args

class SmallNet_pos(nn.Module):
    def __init__(self, num_classes: int = 5, base: int = 8):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, base, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2),
            nn.Conv3d(base, base * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2),
            nn.Conv3d(base * 2, base * 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool3d((6, 6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(base * 4 * 6 * 6 * 6, args.fc_m1),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(args.fc_m1, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def get_net_pos(name: str, nb_cls: int):
    if name == 'cnn3fc1':
        net = SmallNet_pos(num_classes=5)
    else:
        raise Exception('wrong net name', name)
    net_parameters = futil.count_parameters(net)
    net_parameters = str(net_parameters // 1024 // 1024)  # convert to M
    log_dict['net_parameters'] = net_parameters

    return net


def load_data_of_pats(dir_pats: Union[List, np.ndarray], label_file: str):
    df_excel = pd.read_excel(label_file, engine='openpyxl')
    df_excel = df_excel.set_index('PatID')
    x, y = [], []
    for dir_pat in dir_pats:
        x_pat, y_pat = load_data_5labels(dir_pat, df_excel)
        x.append(x_pat)
        y.append(y_pat)
    return x, y


def load_data_5labels(dir_pat: str, df_excel: pd.DataFrame) -> Tuple[str, np.ndarray]:
    data_name = dir_pat
    idx = int(dir_pat.split('Pat_')[-1][:3])
    data_label = []
    for level in [1, 2, 3, 4, 5]:
        y = df_excel.at[idx, 'L' + str(level) + '_pos']
        data_label.append(y)
    return data_name, np.array(data_label)


class SScScoreDataset(Dataset):
    """SSc scoring dataset."""

    def __init__(self, data_x_names: Sequence, world_list: Sequence, index: Sequence = None, transform=None):

        self.data_x_names, self.world_list = np.array(data_x_names), np.array(world_list)

        if index is not None:
            self.data_x_names = self.data_x_names[index]
            self.world_list = self.world_list[index]
        print('loading data ...')
        self.data_x = [futil.load_itk(x, require_ori_sp=True) for x in self.data_x_names]
        self.data_x_np = [i[0] for i in self.data_x]  # shape order: z, y, x
        normalize0to1 = ScaleIntensityRange(a_min=-1500.0, a_max=1500.0, b_min=0.0, b_max=1.0, clip=True)
        self.data_x_np = [normalize0to1(x_np) for x_np in self.data_x_np]
        # scale data to 0~1, it's convinent for future transform during dataloader
        self.data_x_or_sp = [[i[1], i[2]] for i in self.data_x]
        self.ori = np.array([i[1] for i in self.data_x])  # shape order: z, y, x
        self.sp = np.array([i[2] for i in self.data_x])  # shape order: z, y, x
        self.y = []
        for world, ori, sp in zip(self.world_list, self.ori, self.sp):
            labels = [int((level_pos - ori[0]) / sp[0]) for level_pos in world]  # ori[0] is the ori of z axil
            self.y.append(np.array(labels))

        self.data_x_np = [x.astype(np.float32) for x in self.data_x_np]
        self.data_y_np = [y.astype(np.float32) for y in self.y]
        # randomcrop = RandomCropPos()
        # image_, label_ = [], []
        # for image, label in zip(self.data_x_np, self.data_y_np):
        #     i, l = randomcrop(image, label)
        #     image_.append(i)
        #
        #     label_.append(l)
        #
        # noise = RandGaussianNoisePos()
        # for image, label in zip(self.data_x_np, self.data_y_np):
        #     image, label = noise(image, label)
        #

        self.transform = transform

    def __len__(self):
        return len(self.data_y_np)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data = {'image_key': self.data_x_np[idx],
                'label_key': self.data_y_np[idx],
                'world_key': self.world_list[idx],
                'space_key': self.sp[idx],
                'origin_key': self.ori[idx],
                'fpath_key': self.data_x_names[idx]}

        check_aug_effect = 0
        if check_aug_effect:
            def crop_center(img, cropx, cropy):
                y, x = img.shape
                startx = x // 2 - (cropx // 2)
                starty = y // 2 - (cropy // 2)
                return img[starty:starty + cropy, startx:startx + cropx]

            img_before_aug = crop_center(data['image_key'], 512, 512)
            futil.save_itk('aug_before_' + data['fpath_key'].split('/')[-1],
                           img_before_aug, data['origin_key'], data['space_key'], dtype='float')
        # if self.transform:
        #     self.data_xy=[self.transform(image, label) for image, label in zip(self.data_x_np, self.data_y_np)]
        #     self.data_x = [x for x in self.data_xy[0]]
        #     self.data_y = [y for y in self.data_xy[1]]
        #     self.data_x_np = np.array(self.data_x)
        #     self.data_y_np = np.array(self.data_y)
        if self.transform:
            data = self.transform(data)

        if check_aug_effect:
            futil.save_itk('aug_after_' + data['fpath_key'].split('/')[-1],
                           data['image_key'], data['origin_key'], data['space_key'], dtype='float')

        data['image_key'] = torch.as_tensor(data['image_key'])
        data['label_key'] = torch.as_tensor(data['label_key'])

        return data


class AddChannelPosd:
    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        """
        Apply the transform to `img`.
        """
        d = dict(data)
        d['image_key'] = d['image_key'][None]
        return d


class MyNormalizeImagePosd:
    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)

        mean, std = np.mean(d['image_key']), np.std(d['image_key'])
        d['image_key'] = d['image_key'] - mean
        d['image_key'] = d['image_key'] / std
        return d


class Path:
    def __init__(self, id, model_dir=None, check_id_dir=False) -> None:
        self.id = id  # type: int
        self.slurmlog_dir = 'slurmlogs'
        self.model_dir = 'models_pos'
        self.data_dir = 'dataset'

        self.id_dir = os.path.join(self.model_dir, str(int(id)))  # +'_fold_' + str(args.fold)
        if args.mode == 'train' and check_id_dir:  # when infer, do not check
            if os.path.isdir(self.id_dir):  # the dir for this id already exist
                raise Exception('The same id_dir already exists', self.id_dir)

        for dir in [self.slurmlog_dir, self.model_dir, self.data_dir, self.id_dir]:
            if not os.path.isdir(dir):
                os.makedirs(dir)
                print('successfully create directory:', dir)

        self.model_fpath = os.path.join(self.id_dir, 'model.pt')
        self.model_wt_structure_fpath = os.path.join(self.id_dir, 'model_wt_structure.pt')

    def label(self, mode: str):
        return os.path.join(self.id_dir, mode + '_label.csv')

    def pred(self, mode: str):
        return os.path.join(self.id_dir, mode + '_pred.csv')

    def pred_int(self, mode: str):
        return os.path.join(self.id_dir, mode + '_pred_int.csv')

    def pred_world(self, mode: str):
        return os.path.join(self.id_dir, mode + '_pred_world.csv')

    def world(self, mode: str):
        return os.path.join(self.id_dir, mode + '_world.csv')

    def loss(self, mode: str):
        return os.path.join(self.id_dir, mode + '_loss.csv')

    def data(self, mode: str):
        return os.path.join(self.id_dir, mode + '_data.csv')


class RandGaussianNoisePosd:
    def __init__(self, *args, **kargs):
        self.noise = RandGaussianNoise(*args, **kargs)

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        d['image_key'] = self.noise(d['image_key'])
        return d


def shiftd(d, start, z_size, y_size, x_size):
    d['image_key'] = d['image_key'][start[0]:start[0] + z_size, start[1]:start[1] + y_size,
                     start[2]:start[2] + x_size]
    d['label_key'] = d['label_key'] - start[0]  # image is shifted up, and relative position should be down

    d['label_key'][d['label_key'] < 0] = 0  # any position outside the edge would be set as edge
    d['label_key'][d['label_key'] > z_size] = z_size  # any position outside the edge would be set as edge

    return d


class CenterCropPosd:
    def __init__(self, z_size=args.z_size, y_size=args.y_size, x_size=args.x_size):
        self.x_size = x_size
        self.y_size = y_size
        self.z_size = z_size

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        img_shape = d['image_key'].shape
        # print(f'img_shape: {img_shape}')
        assert img_shape[0] >= self.z_size
        assert img_shape[1] >= self.y_size
        assert img_shape[2] >= self.x_size
        middle_point = [shape // 2 for shape in img_shape]
        start = [middle_point[0] - self.z_size // 2, middle_point[1] - self.y_size // 2,
                 middle_point[2] - self.y_size // 2]
        d = shiftd(d, start, self.z_size, self.y_size, self.x_size)

        return d


class RandomCropPosd:
    def __init__(self, z_size=args.z_size, y_size=args.y_size, x_size=args.x_size):
        self.x_size = x_size
        self.y_size = y_size
        self.z_size = z_size

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        # if 'image_key' in data:
        img_shape = d['image_key'].shape  # shape order: z,y x
        assert img_shape[0] >= self.z_size
        assert img_shape[1] >= self.y_size
        assert img_shape[2] >= self.x_size

        valid_range = (img_shape[0] - self.z_size, img_shape[1] - self.y_size, img_shape[2] - self.x_size)
        start = [random.randint(0, v_range) for v_range in valid_range]
        d = shiftd(d, start, self.z_size, self.y_size, self.x_size)
        return d


class ComposePosd:
    """My Commpose to handlllllllle with img and label at the same time.

    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        for t in self.transforms:
            data = t(data)
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


def get_transformd(mode=None):
    """
    The input image data is from 0 to 1.
    :param mode:
    :return:
    """
    xforms = []
    if mode == 'train':
        xforms.extend([RandomCropPosd(), RandGaussianNoisePosd()])
    else:
        xforms.extend([CenterCropPosd()])

    xforms.extend([MyNormalizeImagePosd(), AddChannelPosd()])
    transform = ComposePosd(xforms)

    return transform


def _bytes_to_megabytes(value_bytes):
    return round((value_bytes / 1024) / 1024, 2)


def record_mem_info():
    ''' Memory usage in kB '''

    with open('/proc/self/status') as f:
        memusage = f.read().split('VmRSS:')[1].split('\n')[0][:-3]
    print('int(memusage.strip())')

    return int(memusage.strip())


def record_cpu_info():
    pass


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


def split_dir_pats(data_dir, label_file, ts_id):
    abs_dir_path = os.path.dirname(os.path.realpath(__file__))  # abosolute path of the current .py file
    data_dir = abs_dir_path + "/" + data_dir

    dir_pats = sorted(glob.glob(os.path.join(data_dir, "Pat_*CTimage*.mha")))
    if len(dir_pats)==0:
        dir_pats = sorted(glob.glob(os.path.join(data_dir, "Pat_*", "CTimage*.mha")))

    label_excel = pd.read_excel(label_file, engine='openpyxl')

    # 3 labels for one level
    pats_id_in_excel = pd.DataFrame(label_excel, columns=['PatID']).values
    pats_id_in_excel = [i[0] for i in pats_id_in_excel]
    assert len(dir_pats) == len(pats_id_in_excel)

    # assert the names of patients got from 2 ways
    pats_id_in_dir = [int(path.split('Pat_')[-1][:3]) for path in dir_pats]
    pats_id_in_excel = [int(pat_id) for pat_id in pats_id_in_excel]
    assert pats_id_in_dir == pats_id_in_excel

    ts_dir, tr_vd_dir = [], []
    for id, dir_pt in zip(pats_id_in_dir, dir_pats):
        if id in ts_id:
            ts_dir.append(dir_pt)
        else:
            tr_vd_dir.append(dir_pt)
    return np.array(tr_vd_dir), np.array(ts_dir)


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


def start_run(mode, net, dataloader, epochs, loss_fun, loss_fun_mae, opt, scaler, mypath, epoch_idx,
              valid_mae_best=None):
    print(mode + "ing ......")
    loss_path = mypath.loss(mode)
    if mode == 'train':
        net.train()
    else:
        net.eval()

    batch_idx = 0
    total_loss = 0
    total_loss_mae = 0
    for data in dataloader:

        batch_x = data['image_key'].to(device)
        batch_y = data['label_key'].to(device)

        if amp:
            with torch.cuda.amp.autocast():
                if mode != 'train':
                    with torch.no_grad():
                        pred = net(batch_x)
                else:
                    pred = net(batch_x)
                pred *= data['space_key'][:, 0].reshape(-1, 1).to(device)
                batch_y *= data['space_key'][:, 0].reshape(-1, 1).to(device)

                loss = loss_fun(pred, batch_y)

                loss_mae = loss_fun_mae(pred, batch_y)
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
            pred *= data['space_key'][:, 0].reshape(-1, 1).to(device)
            batch_y *= data['space_key'][:, 0].reshape(-1, 1).to(device)

            loss = loss_fun(pred, batch_y)

            loss_mae = loss_fun_mae(pred, batch_y)

            if mode == 'train':  # update gradients only when training
                opt.zero_grad()
                loss.backward()
                opt.step()

        print(loss.item(), pred[0].clone().detach().cpu().numpy())

        total_loss += loss.item()
        total_loss_mae += loss_mae.item()
        batch_idx += 1

        p1 = threading.Thread(target=record_GPU_info)
        p1.start()

    ave_loss = total_loss / batch_idx
    ave_loss_mae = total_loss_mae / batch_idx
    print("mode:", mode, "loss: ", ave_loss, "loss_mae: ", ave_loss_mae)

    if not os.path.isfile(loss_path):
        with open(loss_path, 'a') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(['step', 'loss', 'mae'])
    with open(loss_path, 'a') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow([epoch_idx, ave_loss, ave_loss_mae])

    if valid_mae_best is not None:
        if ave_loss_mae < valid_mae_best:
            print("old valid loss mae is: ", valid_mae_best)
            print("new valid loss mae is: ", ave_loss_mae)

            valid_mae_best = ave_loss_mae

            print('this model is the best one, save it. epoch id: ', epoch_idx)
            torch.save(net.state_dict(), mypath.model_fpath)
            torch.save(net, mypath.model_wt_structure_fpath)
            print('save_successfully at ', mypath.model_fpath)
        return valid_mae_best
    else:
        return None


def get_column(n, tr_y):
    column = [i[n] for i in tr_y]
    column = [j / 5 for j in column]  # convert labels from [0,5,10, ..., 100] to [0, 1, 2, ..., 20]
    return column


def save_xy(xs, ys, mode, mypath):  # todo: check typing
    with open(mypath.data(mode), 'a') as f:
        writer = csv.writer(f)
        for x, y in zip(xs, ys):
            writer.writerow([x, y])



class MSEHigher(nn.Module):
    """Dice and Xentropy loss"""

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, y_pred, y_true):

        if torch.sum(y_pred) > torch.sum(y_true):
            loss = self.mse(y_pred, y_true)
            print('mormal loss')
        else:
            loss = self.mse(y_pred, y_true) * 5
            print("higher loss")

        return loss


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


def get_mae_best(fpath):
    loss = pd.read_csv(fpath)
    mae = min(loss['mae'].to_list())
    return mae


def sampler_by_disext(tr_y):
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

    # weight = [nb_nonzero/len(world_list) if e[0] == 0 else nb_zero/len(world_list) for e in world_list]
    samples_weight = samples_weight.astype(np.float32)
    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    return sampler


def get_loss(args):
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
        elif args.loss == 'msehigher':
            loss_fun = MSEHigher()
        else:
            raise Exception("loss function is not correct ", args.loss)
    return loss_fun


def prepare_data(mypath, data_dir, label_file, kfold_seed=49, ts_level_nb=240, fold=1, total_folds =4):
    # get data_x names
    kf5 = KFold(n_splits=total_folds, shuffle=True, random_state=kfold_seed)  # for future reproduction

    if ts_level_nb == 240:
        ts_id = [68, 83, 36, 187, 238, 12, 158, 189, 230, 11, 35, 37, 137, 144, 17, 42, 66, 70, 28, 64, 210, 3, 49, 32,
                 236, 206, 194, 196, 7, 9, 16, 19, 20, 21, 40, 46, 47, 57, 58, 59, 60, 62, 116, 117, 118, 128, 134, 216]
        tr_vd_pt, ts_pt = split_dir_pats(data_dir, label_file, ts_id)

        kf_list = list(kf5.split(tr_vd_pt))
        tr_pt_idx, vd_pt_idx = kf_list[fold - 1]
        tr_pt = tr_vd_pt[tr_pt_idx]
        vd_pt = tr_vd_pt[vd_pt_idx]

        tr_x, tr_y = load_data_of_pats(tr_pt, label_file)
        vd_x, vd_y = load_data_of_pats(vd_pt, label_file)
        ts_x, ts_y = load_data_of_pats(ts_pt, label_file)

    else:
        raise Exception('please use correct testing dataset')

    for x, y, mode in zip([tr_x, vd_x, ts_x], [tr_y, vd_y, ts_y], ['train', 'valid', 'test']):
        save_xy(x, y, mode, mypath)
    return tr_x, tr_y, vd_x, vd_y, ts_x, ts_y


def train(id: int):
    mypath = Path(id)

    net = get_net_pos(args.net, 5)

    if args.resample_z == 256:
        data_dir: str = "dataset/LowResolution_fix_size"
    elif args.resample_z == 512:
        data_dir: str = "dataset/LowRes512_192_192"
    elif args.resample_z == 800:
        data_dir: str = "dataset/LowRes800_160_160"
    elif args.resample_z == 1024:
        data_dir: str = "dataset/LowRes1024_128_128"
    else:
        raise Exception("wrong resample_z:" + str(args.resample_z))

    label_file: str = "dataset/SSc_DeepLearning/GohScores.xlsx"
    kfold_seed: int = 49

    log_dict['data_dir'] = data_dir
    log_dict['label_file'] = label_file
    log_dict['data_shuffle_seed'] = kfold_seed

    tr_x, tr_y, vd_x, vd_y, ts_x, ts_y = prepare_data(mypath, data_dir, label_file, kfold_seed=49, ts_level_nb=240, fold=args.fold, total_folds = args.total_folds)
    log_dict['tr_pat_nb'] = len(tr_x)
    log_dict['vd_pat_nb'] = len(vd_x)
    log_dict['ts_pat_nb'] = len(ts_x)

    tr_dataset = SScScoreDataset(data_x_names=tr_x, world_list=tr_y, transform=get_transformd('train'))
    vd_dataset = SScScoreDataset(data_x_names=vd_x, world_list=vd_y, transform=get_transformd('train')) # have comparible learning curve
    ts_dataset = SScScoreDataset(data_x_names=ts_x, world_list=ts_y, transform=get_transformd('train'))

    workers = 5
    log_dict['loader_workers'] = workers
    train_dataloader = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True, num_workers=workers)
    valid_dataloader = DataLoader(vd_dataset, batch_size=args.batch_size, shuffle=False, num_workers=workers)
    # valid_dataloader = train_dataloader
    test_dataloader = DataLoader(ts_dataset, batch_size=args.batch_size, shuffle=False, num_workers=workers)

    net = net.to(device)
    if args.eval_id:
        mypath2 = Path(args.eval_id)
        shutil.copy(mypath2.model_fpath, mypath.model_fpath)  # make sure there is at least one model there
        for mo in ['train', 'valid', 'test']:
            shutil.copy(mypath2.loss(mo), mypath.loss(mo))  # make sure there is at least one model there

        net.load_state_dict(torch.load(mypath.model_fpath, map_location=device))
        valid_mae_best = get_mae_best(mypath2.loss('valid'))
        print(f'load model from {mypath2.model_fpath}, valid_mae_best is {valid_mae_best}')
    else:
        valid_mae_best = 10000

    loss_fun = get_loss(args)
    loss_fun_mae = nn.L1Loss()
    lr = 1e-4
    log_dict['lr'] = lr
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler() if amp else None
    epochs = 1 if args.mode == 'infer' else args.epochs
    for i in range(epochs):  # 20000 epochs
        if args.mode in ['train', 'continue_train']:
            start_run('train', net, train_dataloader, epochs, loss_fun, loss_fun_mae, opt, scaler, mypath,
                      i)
        # run the validation
        valid_mae_best = start_run('valid', net, valid_dataloader, epochs, loss_fun, loss_fun_mae, opt,
                                   scaler, mypath, i, valid_mae_best)
        start_run('test', net, test_dataloader, epochs, loss_fun, loss_fun_mae, opt, scaler, mypath, i)

    record_best_preds(net, train_dataloader, valid_dataloader, test_dataloader, mypath)
    for mode in ['train', 'valid', 'test']:
        if args.eval_id:
            mypath2 = Path(args.eval_id)
            for mo in ['train', 'valid', 'test']:
                shutil.copy(mypath2.data(mo), mypath.data(mo))  # make sure there is at least one model there
                shutil.copy(mypath2.loss(mo), mypath.loss(mo))  # make sure there is at least one model there
                shutil.copy(mypath2.world(mo), mypath.world(mo))  # make sure there is at least one model there
                shutil.copy(mypath2.pred(mo), mypath.pred(mo))  # make sure there is at least one model there
                shutil.copy(mypath2.pred_int(mo), mypath.pred_int(mo))  # make sure there is at least one model there
                shutil.copy(mypath2.pred_world(mo),
                            mypath.pred_world(mo))  # make sure there is at least one model there

        out_dt = confusion.confusion(mypath.world(mode), mypath.pred_world(mode), label_nb=args.z_size, space=1)
        log_dict.update(out_dt)

        icc_ = futil.icc(mypath.world(mode), mypath.pred_world(mode))
        log_dict.update(icc_)


def SlidingLoader(fpath, label, z_size, stride=1, batch_size=1):
    print(f'start load {fpath} for sliding window inference')
    raw_x, ori, sp = futil.load_itk(fpath, require_ori_sp=True)
    normalize0to1 = ScaleIntensityRange(a_min=-1500.0, a_max=1500.0, b_min=0.0, b_max=1.0, clip=True)
    raw_x = normalize0to1(raw_x)
    raw_x = (raw_x - np.mean(raw_x)) / np.std(raw_x)
    assert raw_x.shape[0] > z_size
    ranges = raw_x.shape[0] - z_size
    print(f'ranges: {ranges}')

    batch_patch = []
    batch_new_label = []
    batch_start = []
    i = 0

    start = 0
    while start < ranges:

        if i < batch_size:
            print(f'start: {start}, i: {i}')
            patch: np.ndarray = raw_x[start:start + z_size]  # z, y, z
            patch = patch.astype(np.float32)
            new_label: torch.Tensor = label - start
            patch = patch[None]  # add a channel
            batch_patch.append(patch)
            batch_new_label.append(new_label.numpy())
            batch_start.append(start)

            start += stride
            i += 1

        if start >= ranges or i >= batch_size:
            batch_patch = torch.tensor(np.array(batch_patch))
            batch_new_label = torch.tensor(batch_new_label)
            batch_start = torch.tensor(batch_start)

            yield batch_patch, batch_new_label, batch_start

            batch_patch = []
            batch_new_label = []
            batch_start = []
            i = 0


class Evaluater():
    def __init__(self, net, dataloader, mode, mypath):
        self.net = net
        self.dataloader = dataloader
        self.mode = mode
        self.mypath = mypath

    def run(self):
        for batch_data in self.dataloader:
            for idx in range(len(batch_data['image_key'])):
                sliding_loader = SlidingLoader(batch_data['fpath_key'][idx], batch_data['label_key'][idx],
                                               stride=args.infer_stride, z_size=args.z_size, batch_size=args.batch_size)
                pred_ls = []
                for patch, new_label, start in sliding_loader:
                    batch_x = patch.to(device)

                    if self.mode == 'train':
                        p1 = threading.Thread(target=record_GPU_info)
                        p1.start()

                    if amp:
                        with torch.cuda.amp.autocast():
                            with torch.no_grad():
                                pred = self.net(batch_x)
                    else:
                        with torch.no_grad():
                            pred = self.net(batch_x)

                    pred = pred.cpu().detach().numpy()
                    pred += start.numpy().reshape((-1, 1))  # re organize it to original coordinate
                    pred_ls.append(pred)

                pred_all = np.concatenate(pred_ls, axis=0)

                batch_label: np.ndarray = batch_data['label_key'][idx].cpu().detach().numpy().astype('Int64')
                batch_preds_ave: np.ndarray = np.mean(pred_all, 0)

                batch_preds_int: np.ndarray = batch_preds_ave.astype('Int64')

                batch_preds_world: np.ndarray = batch_preds_ave * batch_data['space_key'][idx][0].item() + \
                                                batch_data['origin_key'][idx][0].item()

                batch_world: np.ndarray = batch_data['world_key'][idx].cpu().detach().numpy()
                head = ['L1', 'L2', 'L3', 'L4', 'L5']
                futil.appendrows_to(self.mypath.label(self.mode), batch_label, head=head)
                futil.appendrows_to(self.mypath.pred(self.mode), batch_preds_ave, head=head)
                futil.appendrows_to(self.mypath.pred_int(self.mode), batch_preds_int, head=head)
                futil.appendrows_to(self.mypath.pred_world(self.mode), batch_preds_world, head=head)
                futil.appendrows_to(self.mypath.world(self.mode), batch_world, head=head)


def record_best_preds(net, train_dataloader, valid_dataloader, test_dataloader, mypath):
    net.load_state_dict(torch.load(mypath.model_fpath, map_location=device))  # load the best weights to do evaluation
    dataloader_dict = {'train': train_dataloader, 'valid': valid_dataloader, 'test': test_dataloader}
    net.eval()
    for mode, dataloader in dataloader_dict.items():
        evaluater = Evaluater(net, dataloader, mode, mypath)
        evaluater.run()


def record_preds(mode, batch_y, pred, mypath, data):
    batch_label = batch_y.cpu().detach().numpy().astype('Int64')
    batch_preds = pred.cpu().detach().numpy()
    batch_preds_int = batch_preds.astype('Int64')
    batch_preds_world = batch_preds_int * data['space_key'] + data['origin_key']

    futil.appendrows_to(mypath.label(mode), batch_label)
    futil.appendrows_to(mypath.pred(mode), batch_preds)
    futil.appendrows_to(mypath.pred_int(mode), batch_preds_int)
    futil.appendrows_to(mypath.pred_world(mode), batch_preds_world)


def fill_running(df: pd.DataFrame):
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
    return df


def correct_type(df: pd.DataFrame):
    for column in df:
        ori_type = type(df[column].to_list()[-1])
        if ori_type is int:
            df[column] = df[column].astype('Int64')  # correct type
    return df


def record_experiment(record_file: str, current_id: Optional[int] = None):
    if current_id is None:  # before the experiment
        lock = FileLock(record_file + ".lock")
        with lock:  # with this lock,  open a file for exclusive access
            with open(record_file, 'a') as csv_file:
                if not os.path.isfile(record_file) or os.stat(record_file).st_size == 0:  # empty?
                    new_id = 1
                    df = pd.DataFrame()
                else:
                    df = pd.read_csv(record_file)
                    last_id = df['ID'].to_list()[-1]
                    new_id = int(last_id) + 1
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
                    index = df.index.to_list()[-1]
                    for key, value in idatime.items():
                        df.at[index + 1, key] = value  #

                df = fill_running(df)
                df = correct_type(df)

                df.to_csv(record_file, index=False)
                shutil.copy(record_file, 'cp_' + record_file)
                df_lastrow = df.iloc[[-1]]
                df_lastrow.to_csv(mypath.id_dir + '/' + record_file, index=False)  # save the record of the current ex
        return new_id
    else:  # at the end of this experiments, find the line of this id, and record the final information
        lock = FileLock(record_file + ".lock")
        with lock:  # with this lock,  open a file for exclusive access
            df = pd.read_csv(record_file)
            index = df.index[df['ID'] == current_id].to_list()
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

            mypath = Path(current_id)  # evaluate old model
            for mode in ['train', 'valid', 'test']:
                lock2 = FileLock(mypath.loss(mode) + ".lock")
                # when evaluating old mode3ls, those files would be copied to new the folder
                with lock2:
                    loss_df = pd.read_csv(mypath.loss(mode))
                    best_index = loss_df['mae'].idxmin()
                    log_dict['metrics_min'] = 'mae'
                    loss = loss_df['loss'][best_index]
                    mae = loss_df['mae'][best_index]
                df.at[index, mode + '_loss'] = round(loss, 2)
                df.at[index, mode + '_mae'] = round(mae, 2)

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
            args_dict.update({'ID': current_id})
            for column in df:
                if column in args_dict.keys() and type(args_dict[column]) is int:
                    df[column] = df[column].astype(float).astype('Int64')  # correct str to float and then int

            df.to_csv(record_file, index=False)
            shutil.copy(record_file, 'cp_' + record_file)
            df_lastrow = df.iloc[[-1]]
            df_lastrow.to_csv(mypath.id_dir + '/' + record_file, index=False)  # save the record of the current ex


def check_time_difference(t1: datetime, t2: datetime):
    t1_date = datetime.datetime(t1.year, t1.month, t1.day, t1.hour, t1.minute, t1.second)
    t2_date = datetime.datetime(t2.year, t2.month, t2.day, t2.hour, t2.minute, t2.second)
    t_elapsed = t2_date - t1_date

    return str(t_elapsed).split('.')[0]  # drop out microseconds


if __name__ == "__main__":


    LogType = Optional[Union[int, float, str]]  # int includes bool
    log_dict: Dict[str, LogType] = {}  # a global dict to store variables saved to log files

    if torch.cuda.is_available():
        device = torch.device("cuda")
        amp = True
    else:
        device = torch.device("cpu")
        amp = False
    log_dict['amp'] = amp

    record_file = 'records_pos.csv'
    id = record_experiment(record_file)
    train(id)
    record_experiment(record_file, current_id=id)

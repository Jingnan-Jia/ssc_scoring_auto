# -*- coding: utf-8 -*-
# @Time    : 3/3/21 12:25 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
import csv
import glob
import itertools
import os
import threading
import time

import SimpleITK as sitk
import argparse

import monai
import numpy as np
import nvidia_smi
import pandas as pd
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import Resize, RandomRotation, RandomHorizontalFlip, RandomVerticalFlip, RandomCrop
from typing import (Dict, List, Tuple, Set, Deque, NamedTuple, IO, Pattern, Match, Text,
                    Optional, Sequence, Union, TypeVar, Iterable, Mapping, MutableMapping, Any)
from torchvision.models import vgg16, vgg19

import torch.nn as nn
from torch.utils.data import WeightedRandomSampler
from set_args import args
from sklearn.model_selection import KFold
import pandas as pd
import datetime
from filelock import FileLock
from varname import nameof
# import streamlit as st
import math
import subprocess
import torchvision.models as models

log_dict = {}  # a global dict to store variables saved to log files


def get_net(name, nb_cls):
    if name=='vgg16':
        net =  models.vgg16(pretrained=False, progress=True, num_classes=nb_cls)
        net.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  # change in_features to 1
    if name=='vgg19':
        net = models.vgg19(pretrained=False, progress=True, num_classes=nb_cls)
        net.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  # change in_features to 1
    if name=='resnet18':
        net = models.resnet18()
    if name=='alex':
        net = models.alexnet()
    if name=='squeezenet':
        net = models.squeezenet1_0()
    if name=='densenet161':
        net = models.densenet161()
    if name=='inception_v3':
        net = models.inception_v3()
    if name=='shufflenet_v2_x1_0':
        net = models.shufflenet_v2_x1_0()
    if name=='resnext50_32x4d':
        net = models.resnext50_32x4d(pretrained=False, progress=True, num_classes=nb_cls)
        net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    if name=='wide_resnet50_2':
        net = models.wide_resnet50_2()
    if name=='mnasnet1_0':
        net = models.mnasnet1_0()
    if name=='resnext101_32x8d':
        net = models.resnext101_32x8d(pretrained=False, progress=True, num_classes=nb_cls)
        net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    return net



def load_itk(filename, require_sp_po=False):
    #     print('start load data')
    # Reads the image using SimpleITK
    if (os.path.isfile(filename)):
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


def load_level_data(data_dir: str, label_file: str, level: int) -> Tuple[List, List]:
    """
    Load the data for the specific level.
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


def load_data_from_dir(data_dir: str, label_file: str) -> Tuple[List, List]:
    """
    The structure of data should be
    - data_dir
      - case1
        - level1_up.mha
        - level1_middle.mha
        - level1_down.mha
        - level2_up.mha
        ...
        - level5_down.mha
      - case2
        ...
      ...
      label_file # an excel file
    :param data_dir:
    :return:
    """
    abs_dir_path = os.path.dirname(os.path.realpath(__file__))  # abosolute path of the current .py file
    data_dir = abs_dir_path + "/" + data_dir
    if args.level == 0:
        level1_x, level1_y = load_level_data(data_dir, label_file, level=1)
        level2_x, level2_y = load_level_data(data_dir, label_file, level=2)
        level3_x, level3_y = load_level_data(data_dir, label_file, level=3)
        level4_x, level4_y = load_level_data(data_dir, label_file, level=4)
        level5_x, level5_y = load_level_data(data_dir, label_file, level=5)

        level_x = list(itertools.chain(level1_x, level2_x, level3_x, level4_x, level5_x));
        level_y = list(itertools.chain(level1_y, level2_y, level3_y, level4_y, level5_y))
    else:
        level_x, level_y = load_level_data(data_dir, label_file, level=args.level)

    return level_x, level_y


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
        self.data_x_np = [load_itk(x) for x in self.data_x_names]
        self.data_x_np = [normalize(x) for x in self.data_x_np]; log_dict['normalize_data'] = True

        self.data_x_np = [x.astype(np.float32) for x in self.data_x_np]
        self.data_y_np = [y.astype(np.float32) for y in self.data_y_list]
        self.data_x = [torch.as_tensor(x) for x in self.data_x_np]
        self.data_y = [torch.as_tensor(y) for y in self.data_y_np]
        self.transform = transform

    def __len__(self):
        return len(self.data_y)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.data_x[idx]
        label = self.data_y[idx]

        if self.transform:
            image = self.transform(image)

        return (image, label)


class AddChannel:
    def __call__(self, img):
        """
        Apply the transform to `img`.
        """
        return img[None]

class Path():
    def __init__(self, id):
        self.id = id
        self.slurmlog_dir = 'slurmlogs'
        self.model_dir = 'models'
        self.data_dir = 'dataset'

        self.id_dir = os.path.join(self.model_dir, str(int(id)), 'fold_'+str(args.fold))

        for dir in [self.slurmlog_dir, self.model_dir, self.data_dir, self.id_dir]:
            if not os.path.isdir(dir):
                os.makedirs(dir)
                print('successfully create directory:', dir)

        self.model_fpath = os.path.join(self.id_dir, 'model.pt')
        self.train_batch_label = os.path.join(self.id_dir, 'train_batch_label.csv')
        self.train_batch_preds = os.path.join(self.id_dir, 'train_batch_preds.csv')
        self.train_batch_preds_int = os.path.join(self.id_dir, 'train_batch_preds_int.csv')
        self.train_batch_preds_end5 = os.path.join(self.id_dir, 'train_batch_preds_end5.csv')

        self.valid_batch_label = os.path.join(self.id_dir, 'valid_batch_label.csv')
        self.valid_batch_preds = os.path.join(self.id_dir, 'valid_batch_preds.csv')
        self.valid_batch_preds_int = os.path.join(self.id_dir, 'valid_batch_preds_int.csv')
        self.valid_batch_preds_end5 = os.path.join(self.id_dir, 'valid_batch_preds_end5.csv')
        self.train_loss = os.path.join(self.id_dir, 'train_loss.csv')
        self.valid_loss = os.path.join(self.id_dir, 'valid_loss.csv')

def get_transform():
    Rotation = 20
    patch_size = 480
    image_size = 512
    vertflip = 0.5
    horiflip = 0.5

    xforms = [

        # Spacingd(keys, pixdim=(self.tsp_xy, self.tsp_xy, self.tsp_z), mode=("bilinear", "nearest")[: len(keys)]),
        AddChannel(),
        Resize(image_size),
        # RandomAffine(degrees=20, scale=(0.8, 1.2),
        # RandomCrop(patch_size),
        # CenterCrop(patch_size),
        # FiveCrop(patch_size), # may lead to more output
        RandomRotation(Rotation),
        RandomHorizontalFlip(p=horiflip),
        RandomVerticalFlip(p=vertflip),
        # ScaleIntensityRanged(keys[0], a_min=-1000.0, a_max=500.0, b_min=0.0, b_max=1.0, clip=True),
        # RandGaussianNoised(keys[0], prob=0.3, std=0.01),
        monai.transforms.RandGaussianNoise()
    ]
    transform = transforms.Compose(xforms)
    global log_dict
    log_dict['RandomVerticalFlip'] = vertflip
    log_dict['RandomHorizontalFlip'] = horiflip
    log_dict['RandomRotation'] = Rotation
    log_dict['image_size'] = image_size
    # log_dict['RandomCrop'] = patch_size
    log_dict['RandGaussianNoise'] = 0.1


    return transform

def _bytes_to_megabytes(bytes):
    return round((bytes/1024)/1024,2)

def record_GPU_info():
    if args.outfile:
        jobid_gpuid = args.outfile.split('-')[-1]
        tmp_split = jobid_gpuid.split('_')[-1]
        if len(tmp_split)==2:
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
        writer.writerows(data)


def train(id):
    mypath = Path(id)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        amp = True
    else:
        device = "cpu"
        amp = False
    log_dict['amp'] = amp

    net = get_net(args.net, 3)

    transform = get_transform()



    # get data_x names
    data_x_names, data_y_list = load_data_from_dir(data_dir="dataset/SSc_DeepLearning",
                                                   label_file="dataset/SSc_DeepLearning/GohScores.xlsx")
    kf5 = KFold(n_splits=5, shuffle=True, random_state=42)  # for future reproduction
    log_dict['data_shuffle'] = True
    log_dict['data_shuffle_seed'] = 42
    kf_list = list(kf5.split(data_x_names))
    train_index, valid_index = kf_list[args.fold - 1]
    log_dict['train_nb'] = len(train_index)
    log_dict['valid_nb'] = len(valid_index)
    log_dict['train_index_head'] = train_index[:20]
    log_dict['valid_index_head'] = valid_index[:20]
    # train_index = train_index[:300]; log_dict['train_nb'] = 300
    # valid_index = train_index[:300]; log_dict['valid_nb'] = 300

    if args.sampler:
        disext_list = []
        for sample in data_y_list:
            disext_list.append(sample[0])
        disext_np = np.array(disext_list)
        disext_unique = np.unique(disext_np)
        class_sample_count = np.array([len(np.where(disext_np == t)[0]) for t in disext_unique])
        weight = 1. / class_sample_count
        disext_unique_list = list(disext_unique)
        samples_weight = np.array([weight[disext_unique_list.index(t)] for t in disext_np])

        # weight = [nb_nonzero/len(data_y_list) if e[0] == 0 else nb_zero/len(data_y_list) for e in data_y_list]
        samples_weight = samples_weight.astype(np.float32)[train_index]
        samples_weight = torch.from_numpy(samples_weight)
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    else:
        sampler = None

    train_dataset = SScScoreDataset(data_x_names, data_y_list, index=train_index, transform=transform)
    valid_dataset = SScScoreDataset(data_x_names, data_y_list, index=valid_index, transform=transform)

    batch_size = 10; log_dict['batch_size'] = batch_size
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, sampler=sampler)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    log_dict['loader_workers'] = 8
    net = net.to(device)
    loss_fun = nn.MSELoss(); log_dict['loss_fun'] = 'MSE'
    loss_fun_mae = nn.L1Loss();

    lr = 1e-4; log_dict[nameof(lr)] = lr
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    if amp:
        scaler = torch.cuda.amp.GradScaler()

    valid_mae_best = 1000
    if args.mode == 'train' or args.eval_id is None:
        epochs = args.epochs
    else:
        epochs = 1
    for i in range(epochs):  # 20000 epochs
        print('Training ...')
        if args.mode != 'train':
            net.load_state_dict(torch.load(mypath.model_fpath, map_location=device))
        else:
            net.train()
            batch_idx = 0
            train_loss = 0
            train_loss_mae = 0
            train_loss_mae_end5 = 0
            for batch_x, batch_y in train_dataloader:

                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                if amp:
                    with torch.cuda.amp.autocast():
                        pred = net(batch_x)
                        loss = loss_fun(pred, batch_y)
                        loss_mae = loss_fun_mae(pred, batch_y)
                        pred_end5 = torch.round(pred / 5) * 5
                        loss_mae_end5 = loss_fun_mae(pred_end5, batch_y)
                else:
                    pred = net(batch_x)
                    loss = loss_fun(pred, batch_y)
                    loss_mae = loss_fun_mae(pred, batch_y)
                    pred_end5 = torch.round(pred / 5) * 5
                    loss_mae_end5 = loss_fun_mae(pred_end5, batch_y)

                opt.zero_grad()

                if amp:
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    opt.step()

                print(loss.item())

                train_loss += loss.item()
                train_loss_mae += loss_mae.item()
                train_loss_mae_end5 += loss_mae_end5.item()
                batch_idx += 1

                p1 = threading.Thread(target=record_GPU_info)
                p1.start()

                if i == epochs-1:  # final epoch
                    batch_label = batch_y.cpu().detach().numpy().astype(int)
                    batch_preds = pred.cpu().detach().numpy()
                    batch_preds_int = batch_preds.astype(int)
                    batch_preds_end5 = np.rint(batch_preds_int / 5) * 5
                    batch_preds_end5 = batch_preds_end5.astype(int)
                    appendrows_to(mypath.train_batch_label, batch_label)
                    appendrows_to(mypath.train_batch_preds, batch_preds)
                    appendrows_to(mypath.train_batch_preds_end5, batch_preds_end5)

                    # with open(mypath.train_batch_preds, 'a') as f:
                    #     np.savetxt(mypath.train_batch_preds, pred.cpu().detach().numpy().astype(int), fmt='%i', delimiter=",")
                    # with open(mypath.train_batch_label, 'a') as f:
                    #     np.savetxt(mypath.train_batch_label, batch_y.cpu().detach().numpy().astype(int), fmt='%i', delimiter=",")

            train_loss = train_loss / batch_idx
            train_loss_mae = train_loss_mae / batch_idx
            train_loss_mae_end5 = train_loss_mae_end5 / batch_idx
            print("train loss: ", train_loss, "train loss_mae: ", train_loss_mae, "train_loss_mae_end5: ", train_loss_mae_end5)

            if not os.path.isfile(mypath.train_loss):
                with open(mypath.train_loss, 'a') as csv_file:
                    writer = csv.writer(csv_file, delimiter=',')
                    writer.writerow(['step', 'loss', 'mae', 'mae_end5'])
            with open(mypath.train_loss, 'a') as csv_file:
                writer = csv.writer(csv_file, delimiter=',')
                writer.writerow([i, train_loss, train_loss_mae, train_loss_mae_end5])

        net.eval()
        print('Evaluating ...')
        valid_loss = 0
        valid_loss_mae = 0
        valid_loss_mae_end5 = 0
        batch_idx = 0
        with torch.no_grad():
            for batch_x, batch_y in valid_dataloader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                if amp:
                    with torch.cuda.amp.autocast():
                        pred = net(batch_x)
                        loss = loss_fun(pred, batch_y)
                        loss_mae = loss_fun_mae(pred, batch_y)
                        pred_end5 = torch.round(pred / 5) * 5
                        loss_mae_end5 = loss_fun_mae(pred_end5, batch_y)

                else:
                    pred = net(batch_x)
                    loss = loss_fun(pred, batch_y)
                    loss_mae = loss_fun_mae(pred, batch_y)
                    pred_end5 = torch.round(pred / 5) * 5
                    loss_mae_end5 = loss_fun_mae(pred_end5, batch_y)

                valid_loss += loss.item()
                valid_loss_mae += loss_mae.item()
                valid_loss_mae_end5 += loss_mae_end5.item()

                batch_idx += 1
                # if batch_idx in [1,2,3,4,5]:
                if i == epochs-1:  # final epoch
                    batch_label = batch_y.cpu().detach().numpy().astype(int)
                    batch_preds = pred.cpu().detach().numpy()
                    batch_preds_int = batch_preds.astype(int)
                    batch_preds_end5 = np.rint(batch_preds/5) * 5
                    batch_preds_end5 = batch_preds_end5.astype(int)
                    appendrows_to(mypath.valid_batch_label, batch_label)
                    appendrows_to(mypath.valid_batch_preds, batch_preds)
                    appendrows_to(mypath.valid_batch_preds_int, batch_preds_int)
                    appendrows_to(mypath.valid_batch_preds_end5, batch_preds_end5)


        valid_loss = valid_loss / batch_idx
        valid_loss_mae = valid_loss_mae / batch_idx
        valid_loss_mae_end5 = valid_loss_mae_end5 / batch_idx
        print("valid loss: ", valid_loss, "valid loss_mae: ", valid_loss_mae, "valid_loss_mae_end5: ", valid_loss_mae_end5)
        if not os.path.isfile(mypath.valid_loss):
            with open(mypath.valid_loss, 'a') as csv_file:
                writer = csv.writer(csv_file, delimiter=',')
                writer.writerow(['step', 'loss', 'mae', 'mae_end5'])
        with open(mypath.valid_loss, 'a') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow([i, valid_loss, valid_loss_mae, valid_loss_mae_end5])
        if args.mode == 'train':
            if valid_loss_mae < valid_mae_best:
                valid_mae_best = valid_loss_mae
                print('this model is the best one, save it.')
                torch.save(net.state_dict(), mypath.model_fpath)
                print('save_successfully!')


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

                date = datetime.date.today().strftime("%Y-%m-%d")
                time = datetime.datetime.now().time().strftime("%H:%M:%S")
                # row = [new_id, date, time, ]
                idatime = {'ID': new_id, 'start_date':date, 'start_time': time}

                args_dict = vars(args)
                idatime.update(args_dict)
                if len(df) == 0:
                    df = pd.DataFrame([idatime])  # need a [] , or need to assign the index for df
                else:
                    for key, value in idatime.items():
                        df.at[new_id-1, key] = value  #
                    # df = df.append(idatime, ignore_index=True)  # would change the dtype of the whole column

                if args.mode == 'train':
                    for index, row in df.iterrows():
                        if 'State' not in list(row.index) or row['State'] in [None, np.nan, 'RUNNING']:
                            try:
                                jobid = row['outfile'].split('-')[-1].split('_')[0]  # extract job id from outfile name
                                seff = os.popen('seff ' + jobid)  # get job information
                                for line in seff.readlines():
                                    line = line.split(': ')  # must have space to be differentiated from time format 00:12:34
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
                        df[column] = df[column].astype(pd.Int64Dtype())  # correct type
                    # except:
                    #     pass
                # df = df.replace(-999, np.nan)
                df.to_csv(record_file, index=False)
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
            t1 = datetime.datetime.strptime(df['start_date'][index] +' ' + df['start_time'][index], f)
            t2 = datetime.datetime.strptime(df['end_date'][index] +' ' + df['end_time'][index], f)
            elapsed_time = check_time_difference(t1, t2)
            df.at[index, 'elapsed_time'] = elapsed_time

            mypath = Path(id)
            lock2 = FileLock(mypath.valid_loss + ".lock")
            with lock2:
                loss_df = pd.read_csv(mypath.valid_loss)
                best_index = loss_df['mae_end5'].idxmin(); log_dict['valid_metrics_min'] = 'mae_end5'
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
                
            df.at[index, 'valid_loss'] = round(valid_loss, 2)
            df.at[index, 'valid_mae'] = round(valid_mae, 2)
            df.at[index, 'valid_mae_end5'] = round(valid_mae_end5, 2)
            df.at[index, 'train_loss'] = round(train_loss, 2)
            df.at[index, 'train_mae'] = round(train_mae, 2)
            df.at[index, 'train_mae_end5'] = round(train_mae_end5, 2)

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
                    df[key] = df[key].astype(pd.Int64Dtype())

            for column in df:
                if type(df[column].to_list()[-1]) is int:
                    df[column] = df[column].astype(pd.Int64Dtype())  # correct type

            args_dict = vars(args)
            args_dict.update({'ID':id})
            for column in df:
                if column in args_dict.keys() and type(args_dict[column]) is int:
                    df[column] = df[column].astype(float).astype(int)  # correct str to float and then int

            df.to_csv(record_file, index=False)

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
    else:
        record_file = 'record_infer.csv'
        record_experiment(record_file)
        id = args.eval_id
    train(id)
    record_experiment(record_file, id=id)
    print('finish this experiments! ')




# -*- coding: utf-8 -*-
# @Time    : 3/21/21 10:15 PM
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
from torchvision.transforms import Resize, RandomRotation, RandomHorizontalFlip, RandomVerticalFlip
from varname import nameof
import confusion
import monai
from monai.transforms import AddChannel, Compose, RandRotate90, Resize, ScaleIntensity, ToTensor


log_dict = {}  # a global dict to store variables saved to log files

from set_args_3D_posi import args

class Path():
    def __init__(self, id, check_id_dir=False):
        self.id = id
        self.slurmlog_dir = 'slurmlogs'
        self.model_dir = 'models'
        self.data_dir = 'dataset'

        self.id_dir = os.path.join(self.model_dir, str(int(id)), 'fold_' + str(args.fold))
        if check_id_dir:
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

        self.valid_batch_label = os.path.join(self.id_dir, 'valid_batch_label.csv')
        self.valid_batch_preds = os.path.join(self.id_dir, 'valid_batch_preds.csv')
        self.valid_batch_preds_int = os.path.join(self.id_dir, 'valid_batch_preds_int.csv')
        self.valid_batch_preds_end5 = os.path.join(self.id_dir, 'valid_batch_preds_end5.csv')

        self.test_batch_label = os.path.join(self.id_dir, 'test_batch_label.csv')
        self.test_batch_preds = os.path.join(self.id_dir, 'test_batch_preds.csv')
        self.test_batch_preds_int = os.path.join(self.id_dir, 'test_batch_preds_int.csv')
        self.test_batch_preds_end5 = os.path.join(self.id_dir, 'test_batch_preds_end5.csv')

        self.train_loss = os.path.join(self.id_dir, 'train_loss.csv')
        self.valid_loss = os.path.join(self.id_dir, 'valid_loss.csv')
        self.test_loss = os.path.join(self.id_dir, 'test_loss.csv')



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

def prepare_data():
    # get data_x names
    data_dir = "dataset/SSc_DeepLearning"
    label_file = "dataset/SSc_DeepLearning/GohScores.xlsx"
    log_dict['data_dir'] = data_dir
    log_dict['label_file'] = label_file
    pat_names = get_dir_pats(data_dir, label_file)
    pat_names = np.array(pat_names)
    ts_pat_names = pat_names[-args.nb_test:]
    tr_vd_pat_names = pat_names[:-args.nb_test]
    kf5 = KFold(n_splits=5, shuffle=True, random_state=42)  # for future reproduction
    log_dict['data_shuffle'] = True
    log_dict['data_shuffle_seed'] = 42
    kf_list = list(kf5.split(tr_vd_pat_names))
    tr_pat_idx, vd_pat_idx = kf_list[args.fold - 1]

    log_dict['train_nb'] = len(tr_pat_idx)
    log_dict['valid_nb'] = len(vd_pat_idx)
    log_dict['test_nb'] = len(ts_pat_names)

    log_dict['train_index_head'] = tr_pat_idx[:20]
    log_dict['valid_index_head'] = vd_pat_idx[:20]

    tr_pat_names = tr_vd_pat_names[tr_pat_idx]
    vd_pat_names = tr_vd_pat_names[vd_pat_idx]

    tr_x, tr_y = load_data_of_pats(tr_pat_names, label_file)
    vd_x, vd_y = load_data_of_pats(vd_pat_names, label_file)
    ts_x, ts_y = load_data_of_pats(ts_pat_names, label_file)

    return tr_x, tr_y, vd_x, vd_y, ts_x, ts_y


def load_data_3D(dir_pat: str, df_excel: pd.DataFrame) -> Tuple[List, List]:
    """
    Load the data for the specific level.
    :param df_excel:
    :param dir_pat:
    :param level:
    :return:
    """
    x = glob.glob(os.path.join(dir_pat, "CTimage*"))  # x is a list with 1 element

    idx = int(dir_pat.split('/')[-1].split('Pat_')[-1])
    y = []
    for i in [1,2,3,4,5]:
        y.append(df_excel.at[idx, 'L' + str(i) + '_pos'])  # x is a list with 5 element

    return x, y

def load_data_of_pats(dir_pats: List, label_file: str):
    df_excel = pd.read_excel(label_file, engine='openpyxl')
    df_excel = df_excel.set_index('PatID')
    x, y = [], []
    for dir_pat in dir_pats:
        # print(dir_pat)
        x_pat, y_pat = load_data_3D(dir_pat, df_excel)
        x.extend(x_pat)
        y.extend(y_pat)
    return x, y

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
        self.data_x_np = [normalize(x) for x in self.data_x_np]
        log_dict['normalize_data'] = True

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


def train(id):
    mypath = Path(id)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        amp = True
    else:
        device = "cpu"
        amp = False
    log_dict['amp'] = amp
    transforms = Compose([ScaleIntensity(), AddChannel(), Resize((96, 96, 96)), ToTensor()])

    net = monai.networks.nets.densenet.densenet121(spatial_dims=3, in_channels=1, out_channels=1).to(device)
    tr_x, tr_y, vd_x, vd_y, ts_x, ts_y = prepare_data()

    tr_dataset = SScScoreDataset(tr_x, tr_y, transform=transform)
    vd_dataset = SScScoreDataset(vd_x, vd_y, transform=transform)
    ts_dataset = SScScoreDataset(ts_x, ts_y, transform=transform)

    batch_size = 10
    log_dict['batch_size'] = batch_size
    workers = 14
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
        loss_fun = nn.MSELoss()  # for regression task
        log_dict['loss_fun'] = 'MSE'
    loss_fun_mae = nn.L1Loss()

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

    confusion.confusion(mypath.train_batch_label, mypath.train_batch_preds_end5)
    confusion.confusion(mypath.valid_batch_label, mypath.valid_batch_preds_end5)
    confusion.confusion(mypath.test_batch_label, mypath.test_batch_preds_end5)




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

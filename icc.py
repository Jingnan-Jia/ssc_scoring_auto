# -*- coding: utf-8 -*-
# @Time    : 4/17/21 8:11 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
import pingouin as pg
import pandas as pd
import numpy as np


dir = "/data/jjia/ssc_scoring/models/614/"

task = 'ssc_score'


def icc(mypath):
    icc_dict = {}
    for mode in ['train', 'valid', 'test']:
        print(mode, '-----------------')

        label = pd.read_csv(mypath.label(mode), header=None)
        pred = pd.read_csv(mypath.pred(mode), header=None)

        if len(label.columns) == 3:
            columns = ['disext', 'gg', 'retp']
        elif task == 'ssc_slice_pick':
            columns = ['L1', 'L2', 'L3', 'L4', 'L5']
        else:
            raise Exception('wrong task')

        label.columns = columns
        pred.columns = columns

        label['ID'] = np.arange(1, len(label) + 1)
        label['rater'] = 'label'

        pred['ID'] = np.arange(1, len(pred) + 1)
        pred['rater'] = 'smallnet'

        data = pd.concat([label, pred], axis=0)

        for column in columns:
            icc = pg.intraclass_corr(data=data, targets='ID', raters='rater', ratings=column).round(2)
            icc = icc.set_index("Type")
            icc = icc.loc['ICC2']['ICC']
            print(icc)
            icc_dict[mode + '_' + column] = icc

    return icc_dict

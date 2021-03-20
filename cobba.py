# -*- coding: utf-8 -*-
# @Time    : 3/19/21 11:22 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
import pandas as pd
import numpy as np
import os

def cobba(label_file, pred_file):
    columns = ['disext', 'gg', 'rept']
    df_label = pd.read_csv(label_file, names=columns)
    df_pred = pd.read_csv(pred_file, names=columns)
    for column in columns:
        label = df_label[column].to_numpy().reshape(-1, 1)
        pred = df_pred[column].to_numpy().reshape(-1, 1)
        pred[pred < 0] = 0
        pred[pred > 100] = 100

        index_label = np.arange(0, 101, 5)
        index_pred = np.arange(0, 101, 5)

        df = pd.DataFrame(0, index=index_label, columns=index_pred)
        # print(df_disext)

        # abs_dif = np.abs(label - pred)
        # scores = np.array([label, pred])
        scores = np.concatenate([label, pred], axis=1)

        for idx in index_label:
            mask = scores[:, 0] == idx
            rows = scores[mask, :]
            unique, counts = np.unique(rows[:, -1], return_counts=True)
            for u, c in zip(unique, counts):
                df.at[idx, u] = c

        basename = os.path.dirname(pred_file)
        prefix = pred_file.split("/")[-1].split("_")[0]
        df.to_csv(basename+'/'+prefix+"_"+column+'_cobba.csv')


if __name__ == "__main__":
    label_file = "/data/jjia/ssc_scoring/models/187/fold_1/test_batch_label.csv"
    pred_file = "/data/jjia/ssc_scoring/models/187/fold_1/test_batch_preds_end5.csv"
    cobba(label_file, pred_file)

# -*- coding: utf-8 -*-
# @Time    : 3/19/21 11:22 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
import glob

import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.metrics import cohen_kappa_score

def get_acc_np(diag_np, total_np):
    acc_ls = []
    for diag, total in zip(diag_np, total_np):
        if diag == 0:
            if total == 0:
                acc_ls.append(0)
            else:
                acc_ls.append(0)
        else:
            acc_ls.append(diag / total)
    acc = np.array(acc_ls)
    return acc


def mae(df):
    # arr = df.to_numpy()
    # labels = df.index.to_numpy().astype(int)
    preds = df.columns.to_numpy().astype(int)
    mae_ls = []
    total_ls = []
    for label, row in df.iterrows():
        row = row.to_numpy()
        abs_err = np.abs(preds - label)
        weighted_err = abs_err * row
        weighted_err = np.sum(weighted_err)
        weighted_err = weighted_err / np.sum(row)
        mae_ls.append(weighted_err)
        total_ls.append(np.sum(row))
    # mae_np = np.mean(, axis=axis, keepdims=keepdims)
    mae_np = np.array(mae_ls)
    total_np = np.array(total_ls)
    return mae_np, total_np

def confusion(label_file, pred_file, label_nb=100, space=5):
    print("Start the save of confsion matrix plot and csv for: ")
    print(label_file)
    print(pred_file)
    df_label = pd.read_csv(label_file)
    if len(df_label.columns)==1:
        columns = ['unknown']
    elif len(df_label.columns)==5:
        columns = ['L1_pos', 'L2_pos', 'L3_pos', 'L4_pos', 'L5_pos']
    else:
        columns = ['disext', 'gg', 'rept']
    df_label = pd.read_csv(label_file, names=columns)
    df_pred = pd.read_csv(pred_file, names=columns)
    out_dt = {}
    for column in columns:
        label = df_label[column].to_numpy().reshape(-1, 1)
        pred = df_pred[column].to_numpy().reshape(-1, 1)
        kappa = cohen_kappa_score(label, pred, weights='linear')
        print(f"weighted kappa for {column} is {kappa}")

        pred[pred < 0] = 0
        pred[pred > label_nb] = label_nb

        index_label = list(range(0, label_nb+1, space))
        index_pred = list(range(0, label_nb+1, space))

        df = pd.DataFrame(0, index=index_label, columns=index_pred)
        scores = np.concatenate([label, pred], axis=1)

        for idx in index_label:
            mask = scores[:, 0] == idx
            rows = scores[mask, :]
            unique, counts = np.unique(rows[:, -1], return_counts=True)
            for u, c in zip(unique, counts):
                df.at[idx, u] = c

        mae_np, total_np = mae(df)

        df.loc[:, 'Total'] = total_np

        diag_np = np.diag(df)

        acc_np = get_acc_np(diag_np, total_np)
        df.loc[:, 'Acc'] = acc_np
        df.loc[:, 'MAE'] = mae_np
        df.loc[0, 'Weighted_kappa'] = kappa
        out_dt['ave_Acc_'+column] = np.mean(acc_np)
        out_dt['ave_MAE_'+column] = np.mean(mae_np)
        out_dt['ave_WKappa_'+column] = kappa

        df.replace(0, np.nan, inplace=True)
        for idx, row in df.iterrows():
            if pd.isna(df.at[idx, 'Acc']) and not pd.isna(df.at[idx, 'Total']):
                df.at[idx, 'Acc'] = 0

        basename = os.path.dirname(pred_file)
        prefix = pred_file.split("/")[-1].split("_")[0]
        df.to_csv(basename + '/' + prefix + "_" + column + '_confusion.csv')


        save_figure = False
        if save_figure:
            fig = plt.figure(figsize=(8,5.5))
            ax = sns.heatmap(df, annot=True, cmap="YlGnBu", fmt='.2f', cbar_kws={"orientation": "horizontal"})  #, cbar=False

            for i in range(len(index_label)):
                ax.add_patch(Rectangle((i+1, i+1), 1, 1, fill=False, edgecolor='blue', ls=':', lw=0.5))

            for text in ax.texts:
                text.set_size(10)
                value = float(text.get_text())
                if value > 99:
                    # pass
                    text.set_size(8)  # number with 3 digits need to show properly
                    # text.set_weight('bold')
                    # text.set_style('italic')

            ax.set_facecolor('xkcd:salmon')
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
            plt.xlabel('Prediction', fontsize=10)  # x-axis label with fontsize 15
            plt.ylabel('Label', fontsize=10)  # y-axis label with fontsize 15

            plt.tight_layout()
            # plt.show()
            fig.savefig(basename+'/'+prefix+"_"+column+'_confusion.png', dpi=fig.dpi)
            plt.close(fig)

        # colormap.set_bad("black")

        print("Finish confsion matrix plot and csv of ", column)
    if len(df_label.columns)>1:
        out_dt['ave_Acc_all'] = 0
        out_dt['ave_MAE_all'] = 0
        out_dt['ave_WKappa_all'] = 0
        for col in columns:
            out_dt['ave_Acc_all'] += out_dt['ave_Acc_' + col]
            out_dt['ave_MAE_all'] += out_dt['ave_MAE_' + col]
            out_dt['ave_WKappa_all'] += out_dt['ave_WKappa_' + col]

    return out_dt

if __name__ == "__main__":
    ids = [528]
    ensemble = False

    for id in ids:
        # mypath = Path(id)
        # confusion(mypath.train_batch_label, mypath.train_batch_preds_end5)
        # confusion(mypath.valid_batch_label, mypath.valid_batch_preds_end5)
        # confusion(mypath.test_batch_label, mypath.test_batch_preds_end5)
        abs_dir_path = os.path.dirname(os.path.realpath(__file__))
        id_dir = abs_dir_path + "/models/" + str(id)

        if not ensemble:
            print(' start comfusion')
            for mode in ['train', 'valid', 'test']:
                label_files = sorted(glob.glob(os.path.join(id_dir, mode + "_label.csv")))
                pred_files = sorted(glob.glob(os.path.join(id_dir, mode + "_pred_int_end5.csv")))
                for label_file, pred_file in zip(label_files, pred_files):
                    confusion(label_file, pred_file)
        else:
            label_files = sorted(glob.glob(os.path.join(id_dir, "*", "test_batch_label.csv")))
            pred_files = sorted(glob.glob(os.path.join(id_dir, "*", "*ensemble.csv")))
            for label_file, pred_file in zip(label_files, pred_files):
                confusion(label_file, pred_file)

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
import statsmodels.api as sm
import numpy as np
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt


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
    # mae_np = np.nanmean(, axis=axis, keepdims=keepdims)
    mae_np = np.array(mae_ls)
    total_np = np.array(total_ls)
    return mae_np, total_np


def confusion(label_file, pred_file, label_nb=100, space=5):
    print("Start the save of confsion matrix plot and csv for: ")
    print(label_file)
    print(pred_file)
    # df_label = pd.read_csv(label_file, header=None)
    # if df_label.iloc[0,0] not in ['L1_pos', 'disext']:
    #     if len(df_label.columns)==5:
    #         columns = ['L1_pos', 'L2_pos', 'L3_pos', 'L4_pos', 'L5_pos']
    #     else:
    #         columns = ['disext', 'gg', 'retp']
    df_label = pd.read_csv(label_file)
    df_pred = pd.read_csv(pred_file)
    if df_label.columns[0] not in ['L1_pos', 'L1', 'disext']:
        df_label = pd.read_csv(label_file, header=None)
        df_pred = pd.read_csv(pred_file, header=None)
        if len(df_label.columns)==5:
            columns = ['L1', 'L2', 'L3', 'L4', 'L5']
        else:
            columns = ['disext', 'gg', 'retp']
        df_label.columns= columns
        df_pred.columns = columns

    out_dt = {}
    fig = plt.figure(figsize=(20,12))
    lower_ls, upper_ls = [], []
    if len(df_label.columns) == 3:
        row_nb, col_nb = 2,2
    elif len(df_label.columns) == 5:
        row_nb, col_nb = 2, 3
    else:
        raise Exception(f'the columns number is not 3 or 5, it is {len(df_label.columns)} ', df_label.columns)

    basename = os.path.dirname(pred_file)
    prefix = pred_file.split("/")[-1].split("_")[0]

    for plot_id, column in enumerate(df_label.columns):
        label = df_label[column].to_numpy().reshape(-1, 1)
        pred = df_pred[column].to_numpy().reshape(-1, 1)

        # bland-altman plot
        ax = fig.add_subplot(row_nb, col_nb, plot_id+1)
        # f, ax = plt.subplots(1, figsize=(8, 5))
        f = sm.graphics.mean_diff_plot(label, pred, ax=ax)
        ax.set_title(column, fontsize=16)
        lower, upper = ax.get_ybound()  # set these plots as the same scale for comparison
        lower_ls.append(lower)
        upper_ls.append(upper)

        # f.savefig(basename + '/' + prefix + "_" + column + '_bland_altman.png')
        # plt.close(f)

        if len(df_label.columns) == 3:
            kappa = cohen_kappa_score(label.astype(int), pred.astype(int), weights='linear')
            print(f"weighted kappa for {column} is {kappa}")

            pred[pred < 0] = 0
            pred[pred > label_nb] = label_nb

            index_label = list(range(0, label_nb + 1, space))
            index_pred = list(range(0, label_nb + 1, space))

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

            if 'valid' in pred_file:
                out_dt['valid_ave_Acc_' + column] = np.nanmean(acc_np)  # there may be some np.nan
                out_dt['valid_ave_MAE_' + column] = np.nanmean(mae_np)
                out_dt['valid_WKappa_' + column] = kappa

            df.replace(0, np.nan, inplace=True)
            for idx, row in df.iterrows():
                if pd.isna(df.at[idx, 'Acc']) and not pd.isna(df.at[idx, 'Total']):
                    df.at[idx, 'Acc'] = 0

            df.to_csv(basename + '/' + prefix + "_" + column + '_confusion.csv')

        # save_figure = False
        # if save_figure:
        #     fig = plt.figure(figsize=(8, 5.5))
        #     ax = sns.heatmap(df, annot=True, cmap="YlGnBu", fmt='.2f',
        #                      cbar_kws={"orientation": "horizontal"})  # , cbar=False
        #
        #     for i in range(len(index_label)):
        #         ax.add_patch(Rectangle((i + 1, i + 1), 1, 1, fill=False, edgecolor='blue', ls=':', lw=0.5))
        #
        #     for text in ax.texts:
        #         text.set_size(10)
        #         value = float(text.get_text())
        #         if value > 99:
        #             # pass
        #             text.set_size(8)  # number with 3 digits need to show properly
        #             # text.set_weight('bold')
        #             # text.set_style('italic')
        #
        #     ax.set_facecolor('xkcd:salmon')
        #     ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        #     ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        #     plt.xlabel('Prediction', fontsize=10)  # x-axis label with fontsize 15
        #     plt.ylabel('Label', fontsize=10)  # y-axis label with fontsize 15
        #
        #     plt.tight_layout()
        #     # plt.show()
        #     fig.savefig(basename + '/' + prefix + "_" + column + '_confusion.png', dpi=fig.dpi)
        #     plt.close(fig)

        # colormap.set_bad("black")

        print("Finish confsion matrix plot and csv of ", column)

    # plot the bland-altman plot of all data
    label = df_label.to_numpy().flatten().reshape(-1, 1)
    pred = df_pred.to_numpy().flatten().reshape(-1, 1)

    ax = fig.add_subplot(row_nb, col_nb, row_nb * col_nb)
    # f, ax = plt.subplots(1, figsize=(8, 5))
    f = sm.graphics.mean_diff_plot(label, pred, ax=ax)
    ax.set_title('All', fontsize=16)
    lower, upper = ax.get_ybound()  # set these plots as the same scale for comparison
    lower_ls.append(lower)
    upper_ls.append(upper)
    lower, upper = min(lower_ls), max(upper_ls)
    for i in range(row_nb * col_nb):
        ax = fig.add_subplot(row_nb, col_nb, i + 1)
        ax.set_ylim(lower, upper)
    f.suptitle(prefix.capitalize() + " Bland-Altman Plot", fontsize=26)
    f.tight_layout()
    f.savefig(basename + '/' + prefix + '_bland_altman.png')
    plt.close(f)

    if ('valid' in pred_file) and (len(df_label.columns) ==3):
        out_dt['valid_ave_Acc_all'] = 0
        out_dt['valid_ave_MAE_all'] = 0
        out_dt['valid_WKappa_all'] = 0
        for col in df_label.columns:
            out_dt['valid_ave_Acc_all'] += out_dt['valid_ave_Acc_' + col]
            out_dt['valid_ave_MAE_all'] += out_dt['valid_ave_MAE_' + col]
            out_dt['valid_WKappa_all'] += out_dt['valid_WKappa_' + col]
        out_dt['valid_ave_Acc_all'] /= len(df_label.columns)
        out_dt['valid_ave_MAE_all'] /= len(df_label.columns)
        out_dt['valid_WKappa_all'] /= len(df_label.columns)

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

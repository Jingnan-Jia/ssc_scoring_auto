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

def confusion(label_file, pred_file):
    print("Start the save of confsion matrix plot and csv for: ")
    print(label_file)
    print(pred_file)
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
        scores = np.concatenate([label, pred], axis=1)

        for idx in index_label:
            mask = scores[:, 0] == idx
            rows = scores[mask, :]
            unique, counts = np.unique(rows[:, -1], return_counts=True)
            for u, c in zip(unique, counts):
                df.at[idx, u] = c

        basename = os.path.dirname(pred_file)
        prefix = pred_file.split("/")[-1].split("_")[0]

        fig = plt.figure(figsize=(7,5.5))
        ax = sns.heatmap(df, annot=True, cmap="YlGnBu", fmt='d')  #, cbar=False

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
        df.to_csv(basename+'/'+prefix+"_"+column+'_confusion.csv')
        print("Finish confsion matrix plot and csv of ", column)




if __name__ == "__main__":
    ids = range(216, 221)
    for id in ids:

        id_dir = "/data/jjia/ssc_scoring/models/" + str(id)
        for mode in ['train', 'valid', 'test']:
            label_files = sorted(glob.glob(os.path.join(id_dir, "*", mode + "_batch_label.csv")))
            pred_files = sorted(glob.glob(os.path.join(id_dir, "*", mode + "_batch_preds_end5.csv")))
            for label_file, pred_file in zip(label_files, pred_files):
                confusion(label_file, pred_file)

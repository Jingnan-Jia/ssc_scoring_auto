import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def extract_label(file_fpath):
    score_disext, score_gg, score_retp = [], [], []
    score_disext_ori, score_gg_ori, score_retp_ori = [], [], []

    with open(file_fpath) as f:
        for row in f:
            divid_flap = "after systhesis, label is  tensor"
            if divid_flap in row:
                score = row.split(divid_flap)[-1]
                score = score[2:-3]  # exclude [( and )]\n
                score = score.split(",")
                try:
                    score_disext.append(int(float(score[0])))
                    score_gg.append(int(float(score[1])))
                    score_retp.append(int(float(score[2])))
                except:
                    pass
            divid_flap2 = "No need for systhesis, label is  tensor"
            if divid_flap2 in row:
                score = row.split(divid_flap2)[-1]
                score = score[2:-3]  # exclude [( and )]\n
                score = score.split(",")
                try:
                    score_disext_ori.append(int(float(score[0])))
                    score_gg_ori.append(int(float(score[1])))
                    score_retp_ori.append(int(float(score[2])))
                except:
                    pass
    fig = plt.figure(figsize=(20, 6))
    for idx, score in enumerate([score_disext, score_gg, score_retp]):
        ax = fig.add_subplot(1, 3, idx+1)
        ax.hist(score, bins=20)
    plt.title("syn")
    plt.show()

    fig = plt.figure(figsize=(20, 6))
    for idx, score in enumerate([score_disext_ori, score_gg_ori, score_retp_ori]):
        ax = fig.add_subplot(1, 3, idx + 1)
        ax.hist(score, bins=20)
    plt.title("ori")
    plt.show()

    fig = plt.figure(figsize=(20, 6))
    score_disext.extend(score_disext_ori)
    score_gg.extend(score_gg_ori)
    score_retp.extend(score_retp_ori)
    for idx, score in enumerate([score_disext,score_gg,score_retp]):
        ax = fig.add_subplot(1, 3, idx + 1)
        ax.hist(score, bins=20)
    plt.title("all")
    plt.show()
    print('finish')



if __name__ == '__main__':
    path = "slurmlogs/slurm-95049_0.out"
    extract_label(file_fpath = path)
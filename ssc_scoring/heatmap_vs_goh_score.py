import medpy
import medpy.metric
import numpy as np
import SimpleITK as sitk
import time
import copy
import matplotlib.pyplot as plt
import glob
import scipy

# This file is used to search/explore the best threshold to let the heat map respresent the Goh Score.

parent_folder = "/home/jjia/data/ssc_scoring/ssc_scoring/results/models/1903/test_data_occlusion_maps_occ_by_healthy"
lung_fpath_ls = sorted(glob.glob(f"{parent_folder}/Pat_*/Level*/lung_mask.npy"))

fig = plt.figure(figsize=(12, 4))
fig_diff = plt.figure(figsize=(12, 4))

title_ls = ['TOT', 'GG', 'RET']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']
y_max, y_max_abs = 0, 0
THRESHOLD = -0.6
abs_sum_ls = []
ratio_ls = [[],[],[]]  # store the ratio of each image for tot, gg and ret.
pred_score_ls = [[],[],[]]  # store the ratio of each image for tot, gg and ret.

for fig_idx, pattern in enumerate(['disext', 'gg', 'rept']):
    map_fpath_ls = sorted(glob.glob(f"{parent_folder}/Pat_*/Level*/{pattern}_ori_label_*_pred_*_mae_diff.npy"))
    print(len(map_fpath_ls),len(lung_fpath_ls) )
    assert len(map_fpath_ls) == len(lung_fpath_ls)

    for map_fpath, lung_fpath in zip(map_fpath_ls, lung_fpath_ls):
        pat_level_1 = map_fpath.split('occlusion_maps_occ_by_healthy')[-1][:16]
        pat_level_2 = lung_fpath.split('occlusion_maps_occ_by_healthy')[-1][:16]
        assert  pat_level_1 == pat_level_2
        pred_score = float(map_fpath.split('pred_')[-1].split('_mae_diff.npy')[0])
        if pred_score <0:
            pred_score = 0
        if pred_score > 100:
            pred_score = 100

        lung_mask = np.load(lung_fpath)
        map = np.load(map_fpath)

        map_ = copy.deepcopy(map)
        map_ = np.where(map_ >= THRESHOLD, 0, 1)
        # map_[map_ < THRESHOLD] = 1
        # map_[map_ >= THRESHOLD] = 0
        area_highted = np.sum(map_)
        area_lung = np.sum(lung_mask)
        ratio = area_highted / area_lung * 100
        ratio_ls[fig_idx].append(ratio)
        pred_score_ls[fig_idx].append(pred_score)

        diff = ratio - pred_score
        row = f"{pattern} ratio: {ratio: .2f}, pred: {pred_score: .2f}, diff: {diff: .2f}"
        # print(row)

ratio_np = np.array(ratio_ls)  # shape: (3, 255)
pred_score_np = np.array(pred_score_ls)  # shape: (3, 255)
diff = ratio_np - pred_score_np
for fig_idx in range(3):
    ax = fig.add_subplot(1, 3, fig_idx + 1)
    ax_diff = fig_diff.add_subplot(1, 3, fig_idx + 1)

    scatter_kwds = {'c': colors[fig_idx], 's':20}
    ratio_ = ratio_np[fig_idx,:]
    pred_score_ = pred_score_np[fig_idx, :]
    diff_ = diff[fig_idx,:]
    # pred_score_, ratio_ = ratio_, pred_score_

    ax.scatter(pred_score_, ratio_, **scatter_kwds)
    ax.set_ylim([0, 100])
    ax.set_xlim([0, 100])
    ax.set_ylabel('Highlighted ratio (%)')
    ax.set_xlabel('Automatic Goh scoring (%)')
    ax.set_title(title_ls[fig_idx])

    ax_diff.scatter(pred_score_, diff_, **scatter_kwds)
    ax_diff.set_ylim([-45, 45])
    ax_diff.set_xlim([0, 100])
    ax_diff.set_ylabel('Highlighted ratio - Automatic Goh scoring (%)')
    ax_diff.set_xlabel('Automatic Goh scoring (%)')
    ax_diff.set_title(title_ls[fig_idx])

    m, b = np.polyfit(pred_score_, ratio_, 1)
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(pred_score_, ratio_)
    print('::::', slope, intercept, r_value, p_value, std_err)
    x_reference = np.array([0, 100])
    print(title_ls[fig_idx], 'linear regression m, b:', m, b)
    print(title_ls[fig_idx], 'linear regression m, b, R^2, p_value:', slope, intercept, r_value ** 2, p_value)

    ax.plot(x_reference, m * x_reference + b, '--', color='gray')  # light gray
    # ax_2.text(0.1, 0.7, '---  Regression line',
    #           ha="left", fontsize='large', transform=ax_2.transAxes)
    ax.text(0.1, 0.7, f'y = {m:.2f}x + {b:.2f}\nR\N{SUPERSCRIPT TWO} = {r_value ** 2: .2f}\nR = {r_value: .2f}\np = {p_value: .5f}',
              ha="left", fontsize='large', transform=ax.transAxes)


fig.show()
fig_diff.show()
fig.savefig(f"heatmap_ratio_vs_goh_score_threshold_-0.6.png")
plt.close(fig)


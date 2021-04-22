# -*- coding: utf-8 -*-
# @Time    : 4/15/21 10:25 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
import jjnutils.util as futil
from scipy import stats
from run import Path, get_net
import pandas as pd
import cv2


def gassuan_blur2d(total_length, smooth_len):
    small_box = np.zeros((total_length - 2 * smooth_len, total_length - 2 * smooth_len))
    big_box = np.pad(small_box, smooth_len, mode='linear_ramp', end_values=1)
    return big_box


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    eval_id = 585
    fold = 3


    pat_name = "Pat_094"
    Level = '2'
    net_name = 'vgg11_bn'
    net = get_net(net_name, 3)
    mypath = Path(eval_id)
    net.load_state_dict(torch.load(mypath.model_fpath, map_location=device))
    net.to(device)

    fill_by_mode = True
    l = 32
    smooth_len = 3
    slide_box = gassuan_blur2d(l, smooth_len)

    img_fpath = mypath.data_dir + "/SSc_DeepLearning/" + pat_name + "/Level" + Level + "_middle_MaskedByLung.mha"
    img_ori_fpath = mypath.data_dir + "/SSc_DeepLearning/" + pat_name + "/Level" + Level + "_middle.mha"

    img_id = int(img_fpath.split('/')[-2].split('_')[-1])
    level = int(img_fpath.split('/')[-1].split('_')[0].split('Level')[-1])
    label_file = "/data/jjia/ssc_scoring/dataset/SSc_DeepLearning/GohScores.xlsx"
    df_excel = pd.read_excel(label_file, engine='openpyxl')
    df_excel = df_excel.set_index('PatID')

    y_disext = df_excel.at[img_id, 'L' + str(level) + '_disext']
    y_gg = df_excel.at[img_id, 'L' + str(level) + '_gg']
    y_retp = df_excel.at[img_id, 'L' + str(level) + '_retp']

    label = np.array([[y_disext, y_gg, y_retp]])  # shape: (1, 3)

    image, origin, space = futil.load_itk(img_fpath, require_ori_sp=True)
    image_len = image.shape[0]
    image[image < -1500] = -1500
    image[image > 1500] = 1500
    image = (image - np.mean(image)) / np.std(image)

    image_min = np.min(image)
    slide_sum_min = image_min * l * l

    attention_map_disext = np.zeros(image.shape)
    attention_map_gg = np.zeros(image.shape)
    attention_map_rept = np.zeros(image.shape)
    attention_maps = [attention_map_disext, attention_map_gg, attention_map_rept]
    attention_names = ['disext_with_attention.jpg', 'gg_with_attention.jpg', 'rept_with_attention.jpg']
    row, column = image_len // l, image_len // l
    for i in range(row):
        for j in range(column):
            print(i, j)
            image_masked = copy.deepcopy(image)
            masked_roi = image_masked[i * l: (i + 1) * l, j * l: (j + 1) * l]
            # masked_roi_small = masked_roi[smooth_len: slide_window - smooth_len, smooth_len: slide_window - smooth_len]
            print(np.sum(masked_roi), slide_sum_min)
            if np.sum(masked_roi) > (slide_sum_min + 1):
                if fill_by_mode:
                    mode, count = stats.mode(masked_roi, axis=None)  # the peak air !
                    image_masked[i * l: (i + 1) * l, j * l: (j + 1) * l] = image_masked[i * l: (i + 1) * l,
                                                                           j * l: (j + 1) * l] * slide_box + mode[0] * (
                                                                                       1 - slide_box)
                else:
                    pass
                    # fill_box = read_image('healthy_image')
                # image[i*8: i*8 + 8, j*8, j*8 +8] = zhongshu(masked_roi)
                image_masked = image_masked[None][None].astype(np.float32)  # 1,1,256,256
                image_masked = torch.tensor(image_masked).to(device)
                pred = net(image_masked)
                pred_minus_label = pred.detach().cpu().numpy() - label  # 3 outputs
                for map, attention_value in zip(attention_maps, pred_minus_label.reshape(-1, )):
                    map[i * l: (i + 1) * l, j * l:(j + 1) * l] = attention_value

    image = (image - np.min(image)) / (np.max(image) - np.min(image) + 0.01)  # rescale to 0-1
    image = (image * 255).astype(int)
    for attention_map, attention_name in zip(attention_maps, attention_names):
        for p_n in ['pos', 'neg']:
            att_map = copy.deepcopy(attention_map)
            if p_n == 'pos':
                att_map[att_map > 0] = 0
                att_map = att_map * (-1)  # convert to positive values to show it via figure
            else:
                att_map[att_map < 0] = 0
            att_map = (att_map - np.min(att_map)) / (np.max(att_map) - np.min(att_map) + 0.01)  # rescale to 0-1
            att_map = (att_map * 255).astype(int)
            # ct_with_attention = image * 0.5 + att_map * 0.5

            hm_po = cv2.applyColorMap(np.uint8(att_map), cv2.COLORMAP_JET)
            slsy_img = 0.3 * hm_po + 0.7 * image.reshape(512, 512, 1)
            score_str = "score_" + str(y_disext) + "_" + str(y_gg) + "_" + str(y_retp)
            fig_fpath = mypath.id_dir + '/' + pat_name + "_level" + Level + score_str + p_n + '_' + attention_name
            cv2.imwrite(fig_fpath, slsy_img)
            cv2.imwrite(fig_fpath.split('.jpg')[0]+'map_only.jpg', hm_po)

            # plt.imshow(ct_with_attention)
            # plt.axis('off')
            #
            # plt.tight_layout()

            # plt.savefig(fig_fpath, bbox_inches='tight')
            # plt.close()

    fig_fpath = mypath.id_dir + '/' + pat_name + "_level" + Level + score_str + '_img_maskedbylung.jpg'
    print(fig_fpath)
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(fig_fpath, bbox_inches='tight')
    plt.close()


    image, origin, space = futil.load_itk(img_ori_fpath, require_ori_sp=True)
    image = (image - np.min(image) )/ (np.max(image) - np.min(image)) * 255
    image = image.astype(int)
    fig_fpath = mypath.id_dir + '/' + pat_name + "_level" + Level + score_str + '_img.jpg'
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(fig_fpath, bbox_inches='tight')
    plt.close()








if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
# @Time    : 3/28/21 2:18 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com

import cv2
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import models
import json
from run import load_itk, get_net, Path, prepare_data, SScScoreDataset, get_transform
from torch.utils.data import Dataset, DataLoader


# 图片预处理
def img_preprocess(img):

    img_out = (img - np.mean(img)) / np.std(img)  # normalize
    img_out = img_out[None]
    img_out = torch.as_tensor(img_out)

    return img_out


# 定义获取梯度的函数
def backward_hook(module, grad_in, grad_out):
    grad_block.append(grad_out[0].detach())


# 定义获取特征图的函数
def farward_hook(module, input, output):
    fmap_block.append(output)


def cam_show_img(img, feature_map, grads, out_dir, idx):
    img = img.numpy()
    _, __, H, W  = img.shape  # (1,1,512,512)
    img = np.resize(img, (H, W, 1))
    cam = np.zeros(feature_map.shape[1:], dtype=np.float32)  # 4
    grads = grads.reshape([grads.shape[0], -1])  # 5
    weights = np.mean(grads, axis=1)  # 6
    for i, w in enumerate(weights):
        cam += w * feature_map[i, :, :]  # 7
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()

    cam = cv2.resize(cam, (W, H))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

    img_jpg = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255

    cam_img = 0.3 * heatmap + 0.7 * img_jpg
    heatmap_img = os.path.join(out_dir, str(idx)+"_cam.jpg")
    path_cam_img = os.path.join(out_dir, str(idx)+"_img_cam.jpg")
    path_img = os.path.join(out_dir,  str(idx)+"_img.jpg")

    cv2.imwrite(heatmap_img, heatmap)
    cv2.imwrite(path_cam_img, cam_img)
    cv2.imwrite(path_img, img_jpg)


def grad_cam(x, y, net, nb_img):
    fmap_block = []
    grad_block = []
    print(y.numpy())

    # forward
    output = net(x)

    # backward
    net.zero_grad()
    loss_fun = torch.nn.MSELoss()
    class_loss = loss_fun(output, y)
    class_loss.backward()

    # 生成cam
    grads_val = grad_block[0].cpu().data.numpy().squeeze()
    fmap = fmap_block[0].cpu().data.numpy().squeeze()
    cam_show_img(x, fmap, grads_val, mypath.id_dir, nb_img)

def saliency_map(x, y, net, nb_img):
    out_ori = net(x)
    out_ori_1 = torch.tensor(o[0] for o in out_ori)
    out_ori_2 = torch.tensor(o[1] for o in out_ori)
    out_ori_3 = torch.tensor(o[2] for o in out_ori)

    x_np = x.clone().detach().numpy()
    y_np = y.clone().detach().numpy()


    map_1 = np.ones_like(x_np)
    map_2 = np.ones_like(x_np)
    map_3 = np.ones_like(x_np)

    x_mean = np.mean(x_np)
    x_std = np.std(x_np)
    x_min = np.min(x_np)
    x_max = np.max(x_np)
    x_0to1 = (x_np - x_min) / (x_max - x_min)
    x_min = np.min(x_np)
    nb_ptchs = 16
    for i in range(nb_ptchs):
        for j in range(nb_ptchs):
            new_x = np.ones_like(x)
            new_x[i*nb_ptchs: (i+1)*nb_ptchs, j*nb_ptchs: (j+1)*nb_ptchs] = 1
            new_x *= x_0to1
            x_out = new_x * (x_max - x_min) + x_min

            out = net(x_out)
            out_np = out.clone().detach().numpy()
            out_1 = np.array([o[0] for o in out_np])
            y_1 = np.array([o[0] for o in y_np])
            out_2 = np.array([o[1] for o in out_np])
            y_2 = np.array([o[1] for o in y_np])
            out_3 = np.array([o[2] for o in out_np])
            y_3 = np.array([o[2] for o in y_np])

            chg_1 = np.sum(np.abs(out_1-out_ori_1))
            chg_2 = np.sum(np.abs(out_2-out_ori_2))
            chg_3 = np.sum(np.abs(out_3-out_ori_3))
            map_1[i*nb_ptchs: (i+1)*nb_ptchs, j*nb_ptchs: (j+1)*nb_ptchs] = chg_1
            map_2[i*nb_ptchs: (i+1)*nb_ptchs, j*nb_ptchs: (j+1)*nb_ptchs] = chg_2
            map_3[i*nb_ptchs: (i+1)*nb_ptchs, j*nb_ptchs: (j+1)*nb_ptchs] = chg_3











if __name__ == '__main__':
    id = 454
    mypath = Path(id)

    tr_x, tr_y, vd_x, vd_y, ts_x, ts_y = prepare_data()
    ts_dataset = SScScoreDataset(ts_x, ts_y, transform=get_transform())
    test_dataloader = DataLoader(ts_dataset, batch_size=10, shuffle=False, num_workers=12)
    test_dataloader = iter(test_dataloader)

    net = get_net('vgg11_bn', 3)
    net.load_state_dict(torch.load(mypath.model_fpath))
    net.eval()  # 8
    print(net)

    nb_img = 0
    while nb_img < 10:
        print(f"nb_img, {nb_img}")
        batch_x, batch_y = next(test_dataloader)
        for x_, y_, idx in zip(batch_x, batch_y, range(10)):
            print(f'idx, {idx}')
            if idx % 3 == 0:
                print(f'idx is okay, {idx}')
                x, y = x_[None], y_[None]
                # grad_cam(x, y, net, nb_img)
                saliency_map(x, y, net, nb_img)


if __name__ == '__main_grad_cam__':
    id = 454
    mypath = Path(id)

    tr_x, tr_y, vd_x, vd_y, ts_x, ts_y = prepare_data()
    ts_dataset = SScScoreDataset(ts_x, ts_y, transform=get_transform())
    test_dataloader = DataLoader(ts_dataset, batch_size=10, shuffle=False, num_workers=12)
    test_dataloader = iter(test_dataloader)

    net = get_net('vgg11_bn', 3)
    net.load_state_dict(torch.load(mypath.model_fpath))
    net.eval()  # 8
    print(net)

    # 注册hook, vgg has features and classifiers
    net.features[3].register_forward_hook(farward_hook)
    net.features[3].register_backward_hook(backward_hook)

    nb_img = 0
    while nb_img < 10:
        print(f"nb_img, {nb_img}")
        batch_x, batch_y = next(test_dataloader)
        for x_, y_, idx in zip(batch_x, batch_y, range(10)):
            print(f'idx, {idx}')
            if idx % 3 == 0:
                print(f'idx is okay, {idx}')

                x, y = x_[None], y_[None]
                grad_cam(x, y, net, nb_img)
                saliency_map(x, y, net, nb_img)

                nb_img += 1

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


# 计算grad-cam并可视化
def cam_show_img_old(img, feature_map, grads, out_dir):
    H, W, _ = img.shape
    cam = np.zeros(feature_map.shape[1:], dtype=np.float32)  # 4
    grads = grads.reshape([grads.shape[0], -1])  # 5
    weights = np.mean(grads, axis=1)  # 6
    for i, w in enumerate(weights):
        cam += w * feature_map[i, :, :]  # 7
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    cam = cv2.resize(cam, (W, H))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    cam_img = 0.3 * heatmap + 0.7 * img

    path_cam_img = os.path.join(out_dir, "cam.jpg")
    cv2.imwrite(path_cam_img, cam_img)


def cam_show_img(img, feature_map, grads, out_dir):
    H, W = img.shape
    cam = np.zeros(feature_map.shape[1:], dtype=np.float32)  # 4
    grads = grads.reshape([grads.shape[0], -1])  # 5
    weights = np.mean(grads, axis=1)  # 6
    for i, w in enumerate(weights):
        cam += w * feature_map[i, :, :]  # 7
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()

    cam = cv2.resize(cam, (W, H))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    cam_img = 0.3 * heatmap + 0.7 * img

    path_cam_img = os.path.join(out_dir, "cam.jpg")
    cv2.imwrite(path_cam_img, cam_img)


if __name__ == '__main__':
    id = 399
    mypath = Path(id)

    tr_x, tr_y, vd_x, vd_y, ts_x, ts_y = prepare_data()
    ts_dataset = SScScoreDataset(ts_x, ts_y, transform=get_transform())
    test_dataloader = DataLoader(ts_dataset, batch_size=10, shuffle=False, num_workers=12)
    test_dataloader = iter(test_dataloader)
    batch_x, batch_y = next(test_dataloader)
    x, y = batch_x[0], batch_y[0]

    # 存放梯度和特征图
    fmap_block = list()
    grad_block = list()

    # 图片读取；网络加载
    # img = load_itk(path_img)
    # img_input = img_preprocess(img)
    net = get_net('vgg11_bn', 3)
    net.load_state_dict(torch.load(mypath.model_fpath))
    net.eval()  # 8
    print(net)

    # 注册hook, vgg has features and classifiers
    net.features[-1].expand3x3.register_forward_hook(farward_hook)
    net.features[-1].expand3x3.register_backward_hook(backward_hook)

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

    # 保存cam图片
    cam_show_img(img, fmap, grads_val, mypath.id_dir)

# -*- coding: utf-8 -*-
# @Time    : 3/6/21 9:58 AM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
# -*- coding: utf-8 -*-

import argparse

parser = argparse.ArgumentParser(description="SSc score prediction.")

parser.add_argument('--mode', choices=('train', 'infer', 'continue_train'), help='mode', type=str, default='train')
parser.add_argument('--eval_id', help='id used for inference, or continue_train', type=int, default=0)

parser.add_argument('--net', choices=('vgg11_bn', 'cnn3fc1', 'cnn2fc1', 'vgg16', 'vgg19', 'resnet18', 'resnext50_32x4d',
                                      'resnext101_32x8d'), help='network name', type=str,
                    default='cnn3fc1')
parser.add_argument('--fc2_nodes', help='the number of nodes of fc2 layer, original is 4096', type=int, default=1024)
parser.add_argument('--fc1_nodes', help='the number of nodes of fc2 layer, original is 4096', type=int, default=1024)
parser.add_argument('--fc_m1', help='the number of nodes of last layer', type=int, default=512)

parser.add_argument('--r_c', choices=('r', 'c'), help='regression or classification?', type=str, default='r')

parser.add_argument('--total_folds', choices=(4, 5), help='4-fold training', type=int, default=4)
parser.add_argument('--fold', choices=(1, 2, 3, 4), help='1 to 4', type=int, default=1)
parser.add_argument('--level', choices=(1, 2, 3, 4, 5, 0), help='level of data, 0 denotes all', type=int, default=0)
parser.add_argument('--sampler', choices=(1, 0), help='if customer sampler?', type=int, default=0)

parser.add_argument('--ts_level_nb', choices=(235, 240), help='if customer sampler?', type=int, default=240)
parser.add_argument('--masked_by_lung', choices=(1, 0), help='if slices are masked by lung masks', type=int, default=0)

image_z = 800  # 512, 800, 1024
if image_z == 256:
    patch_z = 192
    patch_x, patch_y = 256, 256
elif image_z == 512:
    patch_z = 384
    patch_x, patch_y = 192, 192
elif image_z == 800:
    patch_z = 600
    patch_x, patch_y = 160, 160
elif image_z == 1024:
    patch_z = 768
    patch_x, patch_y = 128, 128
else:
    raise Exception("wrong image_z: " + str(image_z))

parser.add_argument('--resample_z', help='resample along z axis', choices=(256, 512, 800, 1024), type=int, default=image_z)
parser.add_argument('--z_size', help='length of patch along z axil ', type=int, default=patch_z)
parser.add_argument('--y_size', help='length of patch along y axil ', type=int, default=patch_x)
parser.add_argument('--x_size', help='length of patch along x axil ', type=int, default=patch_y)

parser.add_argument('--loss', choices=('mse', 'mae', 'smooth_mae', 'mse+mae', 'msehigher'), help='mode', type=str,
                    default='mse')
parser.add_argument('--pretrained', choices=(1, 0), help='pretrained or not', type=int, default=0)
parser.add_argument('--epochs', help='total epochs', type=int, default=2)
parser.add_argument('--weight_decay', help='L2 regularization', type=float, default=0.0)  # must be a float number !
parser.add_argument('--batch_size', help='batch_size', type=int, default=5)
parser.add_argument('--infer_stride', help='infer_stride', type=int, default=4)

parser.add_argument('--outfile', help='output file when running by script instead of pycharm', type=str)
parser.add_argument('--hostname', help='hostname of the server', type=str)
parser.add_argument('--remark', help='comments on this experiment', type=str)

args = parser.parse_args()

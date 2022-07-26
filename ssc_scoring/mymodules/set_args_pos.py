# -*- coding: utf-8 -*-
# @Time    : 3/6/21 9:58 AM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
# -*- coding: utf-8 -*-

import argparse


def get_args():
    parser = argparse.ArgumentParser(description="SSc score prediction.")

    # Common args with set_args.py
    parser.add_argument('--mode', choices=('train', 'infer', 'continue_train'), help='mode', type=str, default='train')
    parser.add_argument('--eval_id', help='id used for inference, or continue_train', type=int, default=0)
    parser.add_argument('--net', choices=('vgg11_3d','vgg16_3d','vgg19_3d', 'r3d_resnet', 'cnn3fc1', 'cnn4fc2', 'cnn5fc2', 'cnn6fc2',
                                          'cnn2fc1', 'cnn3fc2', 'r3d_18'), help='network name', type=str, default='vgg16_3d')
    parser.add_argument('--fc2_nodes', help='the number of nodes of fc2 layer, original is 4096', type=int,
                        default=1024)
    parser.add_argument('--fc1_nodes', help='the number of nodes of fc2 layer, original is 4096', type=int,
                        default=1024)
    parser.add_argument('--base', help='the number of nodes of fc2 layer, original is 4096', type=int,
                        default=8)
    parser.add_argument('--total_folds', choices=(4, 5), help='4-fold training', type=int, default=4)
    parser.add_argument('--fold', choices=(1, 2, 3, 4), help='1 to 4', type=int, default=1)
    parser.add_argument('--valid_period', help='how many epochs between 2 validation', type=int, default=5)
    parser.add_argument('--workers', help='number of workers for dataloader', type=int, default=6)
    parser.add_argument('--ts_level_nb', choices=(235, 240, 250), help='if customer sampler?', type=int, default=250)
    parser.add_argument('--loss', choices=('mse', 'mae', 'smooth_mae', 'mse+mae', 'msehigher'), help='mode', type=str,
                        default='mse')
    parser.add_argument('--pretrained', choices=(1, 0), help='pretrained or not', type=int, default=0)
    parser.add_argument('--epochs', help='total epochs', type=int, default=500)
    parser.add_argument('--weight_decay', help='L2 regularization', type=float,
                        default=0.0001)  # must be a float number !
    parser.add_argument('--batch_size', help='batch_size', type=int, default=4)
    parser.add_argument('--outfile', help='output file when running by script instead of pycharm', type=str, default='None')
    parser.add_argument('--hostname', help='hostname of the server', type=str, default='None')
    parser.add_argument('--remark', help='comments on this experiment', type=str, default='None')
    # Exclusive args
    parser.add_argument('--train_on_level', choices=(1, 2, 3, 4, 5, 0), help='level 0 denotes all', type=int,
                        default=0)
    parser.add_argument('--level_node', choices=(1, 0), help='if network has an extra level node', type=int, default=0)
    parser.add_argument('--kd', choices=('dist', 'transf', 'no'), help='mode', type=str, default='no')
    parser.add_argument('--kd_t_name', choices=('resnet3d_10', 'resnet3d_18', 'resnet3d_34', 'resnet3d_50',
                                                'resnet3d_101', 'resnet3d_152', 'resnet3d_200', 'unet3d'),
                        type=str, default='resnet3d_34')
    parser.add_argument('--infer_2nd', choices=(1, 0), help='', type=int, default=0)
    parser.add_argument('--resample_z', help='resample along z axis', choices=(0, 256, 512, 800, 1024),
                        type=int, default=256)
    parser.add_argument('--z_size', help='length of patch along z axil ', type=int, default=192)
    parser.add_argument('--y_size', help='length of patch along y axil ', type=int, default=256)
    parser.add_argument('--x_size', help='length of patch along x axil ', type=int, default=256)
    parser.add_argument('--infer_stride', help='infer_stride', type=int, default=4)

    args = parser.parse_args()

    if args.level_node == 1:
        args.train_on_level = 0  # use data from all levels to train this network

    if args.resample_z == 256:
        args.z_size = 192
        args.x_size, args.y_size = 256, 256
    elif args.resample_z == 512:
        args.z_size = 384
        args.x_size, args.y_size = 192, 192
    elif args.resample_z == 800:
        args.z_size = 600
        args.x_size, args.y_size = 160, 160
    elif args.resample_z == 1024:
        args.z_size = 192
        args.x_size, args.y_size = 256, 256
    elif args.resample_z == 0:
        args.z_size = 192
        args.x_size, args.y_size = 512, 512
    else:
        raise Exception("wrong resample_z: " + str(args.resample_z))

    if args.x_size == 0 or args.y_size == 0:
        raise Exception("0 x_size or y_size: ")

    return args


if __name__ == "__main__":
    get_args()

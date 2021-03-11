# -*- coding: utf-8 -*-
# @Time    : 3/6/21 9:58 AM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
# -*- coding: utf-8 -*-

import argparse

parser = argparse.ArgumentParser(description="SSc score prediction.")

parser.add_argument('--mode', choices=('train', 'infer'), help='mode', type=str, default='train')
parser.add_argument('--eval_id', help='id used for inference', type=int, default=None)

parser.add_argument('--fold', choices=(1, 2, 3, 4, 5), help='5-fold training', type=int, default=1)
parser.add_argument('--level', choices=(1, 2, 3, 4, 5, 0), help='level of data, 0 denotes all', type=int, default=0)

parser.add_argument('--epochs', help='total epochs', type=int, default=600)
parser.add_argument('--net', choices=('vgg16', 'vgg19', 'resnext50_32x4d'), help='network name', type=str,
                    default='vgg19')
parser.add_argument('--sampler', help='if customer sampler?', type=bool, default=False)

parser.add_argument('--outfile', help='output file when running by script instead of pycharm', type=str,
                    default=None)
parser.add_argument('--hostname', help='hostname of the server', type=str)
parser.add_argument('--remark', help='comments on this experiment', type=str)


args = parser.parse_args()



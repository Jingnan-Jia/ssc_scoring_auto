import collections
import csv
import datetime
import glob
import itertools
import os
import random
import shutil
import threading
import time
from typing import (List, Tuple, Optional, Union, Dict)
from collections import OrderedDict
from tqdm import tqdm

import SimpleITK as sitk
import numpy as np
import nvidia_smi
import pandas as pd
import torch
import torch.nn as nn
# import streamlit as st
import torchvision.models as models
from filelock import FileLock
from monai.transforms import ScaleIntensityRange, RandGaussianNoise
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip, CenterCrop, RandomAffine
import jjnutils.util as futil


import confusion
from set_args import args
import pingouin as pg
import random
import copy

LogType = Optional[Union[int, float, str]]  # int includes bool
log_dict: Dict[str, LogType] = {}  # a global dict to store variables saved to log files

def summary(model, input_size, batch_size=-1, device="cuda"):

    def register_hook(module):

        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
    # print(type(x[0]))

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    print("----------------------------------------------------------------")
    line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
    print(line_new)
    print("================================================================")
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]
        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        print(line_new)

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    print("================================================================")
    print("Total params: {0:,}".format(total_params))
    print("Trainable params: {0:,}".format(trainable_params))
    print("Non-trainable params: {0:,}".format(total_params - trainable_params))
    print("----------------------------------------------------------------")
    print("Input size (MB): %0.2f" % total_input_size)
    print("Forward/backward pass size (MB): %0.2f" % total_output_size)
    print("Params size (MB): %0.2f" % total_params_size)
    print("Estimated Total Size (MB): %0.2f" % total_size)
    print("----------------------------------------------------------------")
    return summary

class ReconNet(nn.Module):
    def __init__(self, reg_net, input_size=512):
        super().__init__()
        self.reg_net = reg_net
        self.features = copy.deepcopy(reg_net.features)  # encoder
        self.decoder = self._build_dec_from_enc()

    def _last_channels(self):
        last_chn = None
        for layer in self.features[::-1]:
            if isinstance(layer, torch.nn.Conv2d):
                last_chn = layer.out_channels
                break
        if last_chn is None:
            raise Exception("No convolution layers at all in regression network")
        return last_chn

    def _build_dec_from_enc(self):
        decoder_ls = []
        in_channels = None  # statement for convtransposed
        last_chns = self._last_channels()

        layer_shapes = summary(self.features, (1, 512, 512))  # ordered dict saving shapes of each layer
        transit_chn = 0
        for layer, (layer_name, layer_shape) in zip(self.features[::-1], reversed(layer_shapes.items())):
            if transit_chn == 0:
                transit_chn = layer_shape['input_shape'][1]

            if isinstance(layer, torch.nn.Conv2d):

                enc_in_channels = layer_shape['input_shape'][1]
                # enc_out_channels = layer_shape['output_shape'][1]

                enc_kernel_size: int = layer.kernel_size[0]  # square kernel, get one of the sizes
                enc_stride: int = layer.stride[0]

                if enc_stride > 1:  # shape is reduced
                    decoder_ls.append(nn.Upsample(scale_factor=enc_stride, mode='bilinear'))
                    decoder_ls.append(nn.Conv2d(transit_chn, enc_in_channels, enc_kernel_size, padding=enc_kernel_size - enc_kernel_size//2-1))
                else:
                    decoder_ls.append(nn.Conv2d(transit_chn, enc_in_channels, enc_kernel_size, padding=enc_kernel_size - enc_kernel_size//2-1))

                decoder_ls.extend([nn.BatchNorm2d(enc_in_channels),
                                   nn.ReLU(inplace=True)])

                transit_chn = enc_in_channels  # new value

            elif isinstance(layer, torch.nn.MaxPool2d):
                decoder_ls.append(nn.Upsample(scale_factor=2, mode='bilinear'))
                decoder_ls.append(nn.Conv2d(transit_chn, transit_chn, 3, padding=1))
                decoder_ls.extend([nn.BatchNorm2d(transit_chn),
                                   nn.ReLU(inplace=True)])

            else:
                pass
        # correct the shape of the final output
        while (isinstance(decoder_ls[-1], nn.ReLU)) or (isinstance(decoder_ls[-1], nn.BatchNorm2d)):
            decoder_ls.pop()
        decoder = nn.Sequential(*decoder_ls)

        class EncDoc(nn.Module):
            def __init__(self, enc, dec):
                super().__init__()
                self.enc = enc
                self.dec = dec
            def forward(self, x):
                out = self.enc(x)
                out = self.dec(out)
                return out


        enc_dec = EncDoc(self.features, decoder)
        enc_dec_layer_shapes = summary(enc_dec, (1, 512, 512))
        input_sz = list(iter(enc_dec_layer_shapes.items()))[0][-1]['input_shape'][-1]
        output_sz = list(iter(enc_dec_layer_shapes.items()))[-1][-1]['input_shape'][-1]
        dif: int = input_sz - output_sz
        if dif > 0:  # the last output of decoder is less than the first output of encoder, need pad
            enc_dec = nn.Sequential(enc_dec,
                                    nn.Upsample(size=input_sz, mode="bilinear"),
                                    nn.Conv2d(1, 1, 3, padding=1))


        # decoder_dict = OrderedDict([(key, value) for key, value in zip(range(len(decoder_ls)), decoder_ls)])
        tmp = summary(enc_dec, (1, 512, 512))
        return enc_dec

    def forward(self, x):
        out = self.enc_dec(x)

        return out

class Cnn2fc1(nn.Module):

    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)

        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(128 * 6 * 6, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


net = Cnn2fc1()
enc_dec_net = ReconNet(net)
model_path = "/data/jjia/ssc_scoring/models/781/model.pt"
torch.load()
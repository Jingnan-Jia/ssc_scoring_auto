# -*- coding: utf-8 -*-
# @Time    : 4/15/21 10:25 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
import torch
import numpy as np
import jjnutils.util as futil
import cv2
import torchvision.models as models
import torch.nn as nn
import os
import pandas as pd


class Path:
    def __init__(self, id, check_id_dir=False) -> None:
        self.id = id  # type: int
        self.slurmlog_dir = 'slurmlogs'
        self.model_dir = 'models'
        self.data_dir = 'dataset'

        self.id_dir = os.path.join(self.model_dir, str(int(id)))  # +'_fold_' + str(args.fold)
        if check_id_dir:  # when infer, do not check
            if os.path.isdir(self.id_dir):  # the dir for this id already exist
                raise Exception('The same id_dir already exists', self.id_dir)

        for dir in [self.slurmlog_dir, self.model_dir, self.data_dir, self.id_dir]:
            if not os.path.isdir(dir):
                os.makedirs(dir)
                print('successfully create directory:', dir)

        self.model_fpath = os.path.join(self.id_dir, 'model.pt')
        self.model_wt_structure_fpath = os.path.join(self.id_dir, 'model_wt_structure.pt')

    def label(self, mode: str):
        return os.path.join(self.id_dir, mode + '_label.csv')

    def pred(self, mode: str):
        return os.path.join(self.id_dir, mode + '_pred.csv')

    def pred_int(self, mode: str):
        return os.path.join(self.id_dir, mode + '_pred_int.csv')

    def pred_end5(self, mode: str):
        return os.path.join(self.id_dir, mode + '_pred_int_end5.csv')

    def loss(self, mode: str):
        return os.path.join(self.id_dir, mode + '_loss.csv')

    def data(self, mode: str):
        return os.path.join(self.id_dir, mode + '_data.csv')


class GradCAM():
    def __init__(self):
        self.pat_name = "Pat_094"
        self.eval_id = 585
        self.Level = '4'
        self.mypath = Path(self.eval_id)
        self.img_fpath = self.mypath.data_dir + "/SSc_DeepLearning/" + self.pat_name + "/Level" + self.Level + "_middle.mha"


        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")


        self.grad_block = []
        self.fmap_block = []
        # 定义获取梯度的函数
    def backward_hook(self, module, grad_in, grad_out):
        self.grad_block.append(grad_out[0].detach())

    # 定义获取特征图的函数
    def farward_hook(self, module, input, output):
        self.fmap_block.append(output)

    def run(self):
        net = models.vgg11_bn(pretrained=0, progress=True)
        net.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  # change in_features to 1
        net.classifier[0] = torch.nn.Linear(in_features=512 * 7 * 7, out_features=1024)
        net.classifier[3] = torch.nn.Linear(in_features=1024, out_features=1024)
        net.classifier[6] = torch.nn.Linear(in_features=1024, out_features=3)

        net.load_state_dict(torch.load(self.mypath.model_fpath, map_location=self.device))
        net.to(self.device)
        net.features[-1].register_forward_hook(self.farward_hook)
        net.features[-1].register_backward_hook(self.backward_hook)
        net.eval()
        opt = torch.optim.Adam(net.parameters(), lr=0.0001)


        image, origin, space = futil.load_itk(self.img_fpath, require_ori_sp=True)
        image_len = image.shape[0]
        image[image < -1500] = -1500
        image[image > 1500] = 1500
        image = (image - np.mean(image)) / np.std(image)
        image = image.astype(np.float32)
        w, h = image.shape
        img = image[None][None]
        img = torch.tensor(img)

        img_id = int(self.img_fpath.split('/')[-2].split('_')[-1])
        level = int(self.img_fpath.split('/')[-1].split('_')[0].split('Level')[-1])
        label_file = "/data/jjia/ssc_scoring/dataset/SSc_DeepLearning/GohScores.xlsx"
        df_excel = pd.read_excel(label_file, engine='openpyxl')
        df_excel = df_excel.set_index('PatID')

        y_disext = df_excel.at[img_id, 'L' + str(level) + '_disext']
        y_gg = df_excel.at[img_id, 'L' + str(level) + '_gg']
        y_retp = df_excel.at[img_id, 'L' + str(level) + '_retp']



        label_all = torch.tensor(np.array([[y_disext, y_gg, y_retp]]).astype(np.float32))
        label_total = torch.tensor(np.array([[y_disext, 0, 0]]).astype(np.float32))
        label_gg = torch.tensor(np.array([[0, y_gg, 0]]).astype(np.float32))
        label_retp = torch.tensor(np.array([[0, 0, y_retp]]).astype(np.float32))

        loss = nn.MSELoss()
        image = (image - image.min()) / (image.max() - image.min())
        image = np.expand_dims(image, -1)
        output = net(img)
        label_str = str(label_all[0, 0].item()) + '_' +str(label_all[0, 1].item()) + '_' +str(label_all[0, 2].item())
        for name, label in zip(['all', 'total', 'gg', 'retp'], [label_all, label_total, label_gg,label_retp ]):
            self.grad_block = []
            print(f"predict: {output.detach().cpu().numpy()}, label: {label.detach().cpu().numpy()}")

            l = loss(output,label )
            opt.zero_grad()  # clear the gradients
            l.backward(retain_graph=True)
            grad = torch.stack(self.grad_block, 0)
            grads_mean = torch.mean(grad, [1, 2])

            cam = torch.zeros(list(self.fmap_block[0].shape))
            for weight, map in zip(grads_mean, self.fmap_block):
                cam += weight * map
            cam = cam[0]
            cam = torch.mean(cam, 0)
            cam = cam.cpu().detach().numpy()


            cam = np.maximum(cam, 0)
            cam = cam / cam.max()
            cam = cv2.resize(cam, (w, h))

            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

            cam_img = 0.3 * heatmap + 0.7 * image*255

            cv2.imwrite('cam_'+self.pat_name+'level_' +Level +name+'.png', cam_img)
        cv2.imwrite('cam_' + self.pat_name + 'level_' + Level + 'label' +  label_str + '.png', image*255)


def gassuan_blur2d(total_length, smooth_len):
    small_box = np.zeros((total_length - 2 * smooth_len, total_length - 2 * smooth_len))
    big_box = np.pad(small_box, smooth_len, mode='linear_ramp', end_values=1)
    return big_box


def main():
    cam = GradCAM()
    cam.run()
    print('finish')








if __name__ == "__main__":
    main()

import torch
from torch.utils.data.dataset import Dataset
import cv2
import pandas as pd
import numpy as np

class MyDataset(Dataset):
    def __init__(self, csv_path, num_primary_color, mode=None):
        self.csv_path = csv_path
        if mode == 'train':
            self.imgs_path = np.array(pd.read_csv(csv_path, header=None)).reshape(-1)[:-10] #csvリストの後ろをvaldataに
            self.palette_list = np.array(pd.read_csv('palette_%d_%s' % (num_primary_color, csv_path), header=None)).reshape(-1, num_primary_color*3)[:-10]
        if mode == 'val':
            self.imgs_path = np.array(pd.read_csv(csv_path, header=None)).reshape(-1)[-10:]
            self.palette_list = np.array(pd.read_csv('palette_%d_%s' % (num_primary_color, csv_path), header=None)).reshape(-1, num_primary_color*3)[-10:]
        if mode == 'test':
            self.imgs_path = np.array(pd.read_csv(csv_path, header=None)).reshape(-1)
            self.palette_list = np.array(pd.read_csv('palette_%d_%s' % (num_primary_color, csv_path), header=None)).reshape(-1, num_primary_color*3)

        self.num_primary_color = num_primary_color

    def __getitem__(self, index):
        img = cv2.imread(self.imgs_path[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose((2,0,1))
        target_img = img/255 # 0~1

        # select primary_color
        primary_color_layers = self.make_primary_color_layers(self.palette_list[index], target_img)

        # to Tensor
        target_img = torch.from_numpy(target_img.astype(np.float32))
        primary_color_layers = torch.from_numpy(primary_color_layers.astype(np.float32))

        return target_img, primary_color_layers # return torch.Tensor

    def __len__(self):
        return len(self.imgs_path)


    def make_primary_color_layers(self, palette_values, target_img):
        '''
         入力：パレットの色の値
         出力：primary_color_layers #out: (bn, ln, 3, h, w)
        '''
        primary_color = palette_values.reshape(self.num_primary_color, 3) / 255 # (ln, 3)
        primary_color_layers = np.tile(np.ones_like(target_img), (self.num_primary_color,1,1,1)) * primary_color.reshape(self.num_primary_color,3,1,1)
        return primary_color_layers

# import
from pathlib import Path
import scipy.io
import torch
from torch.utils.data import Dataset
import os
import numpy as np
from torchvision.transforms.functional import to_tensor, resize
import torch.nn.functional as F
from PIL import Image


# pad psf
#pad_size = (1320-520) // 2

def transform(image):
    image = image.copy()
    image = to_tensor(image)
    return image

def region_of_interest(x):
    return  x[..., 16:784, 16:784]


def load_psf(path):
    psf = scipy.io.loadmat(path)['psf_RGB'].astype(np.float32)
    # print(psf.shape)
    psf = transform(psf)
    # padding = (pad_size, pad_size, pad_size, pad_size)
    # psf = F.pad(psf, padding, mode='constant', value=0)
    # print(psf.shape)
    return psf


class generate_dataset(Dataset):
    def __init__(self, path):
        # txt
        self.data_list = []
        dataset_path = path
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data_list.append(line.strip())

    def __len__(self):

        return len(self.data_list)

    def __getitem__(self, idx):
        '''
        num = 'model_output' + str(idx).zfill(5)+ '.npy'
        diffused = self.diffuser_path / num
        gt = self.gt_path / num
        x = transform(np.load(diffused).astype(np.float32) / 255.)
        # only reconstruct: gt
        y = transform(np.load(gt).astype(np.float32) / 255.)
        '''
        path = self.data_list[idx]
        # name
        img_name = path[-25:-17] + path[-13:-6] + path[-4:]   # train: -25  val: -21
        img = Image.open(path)  # .convert('RGB')
        img = np.array(img)
        img = transform(img.astype(np.float32) / 255.)
        diffused = img[:, :, 1320:2640]
        gt = img[:, :, 0:1320]

        return diffused, gt, img_name


class LenslessLearningCollection:
    def __init__(self, path, isTrain):
        path = Path(path)

        psf_0 = load_psf(path / 'psf.mat').unsqueeze(1)  # unsqueeze(0)
        psf_1 = load_psf(path / '4_nc_even_series_cubic_-20_-20.mat').unsqueeze(1)
        psf_2 = load_psf(path / '4_nc_even_series_cubic_-20_20.mat').unsqueeze(1)
        psf_3 = load_psf(path / '4_nc_even_series_cubic_20_-20.mat').unsqueeze(1)
        psf_4 = load_psf(path / '4_nc_even_series_cubic_20_20.mat').unsqueeze(1)
        psf_5 = load_psf(path / '4_nc_even_series_cubic_-20_0.mat').unsqueeze(1)
        psf_6 = load_psf(path / '4_nc_even_series_cubic_0_-20.mat').unsqueeze(1)
        psf_7 = load_psf(path / '4_nc_even_series_cubic_20_0.mat').unsqueeze(1)
        psf_8 = load_psf(path / '4_nc_even_series_cubic_0_20.mat').unsqueeze(1)
        self.psf = torch.cat((psf_0, psf_1, psf_2, psf_3, psf_4, psf_5, psf_6, psf_7, psf_8), dim=1)

        self.ROI = region_of_interest

        if isTrain:
            path_train = path / 'train_dataset.txt'
            path_val = path / 'val_dataset.txt'
            self.train_dataset = generate_dataset(path_train)
            self.val_dataset = generate_dataset(path_val)
        else:
            path_test = path / 'val_dataset.txt'  #   'train_dataset.txt'
            self.test_dataset = generate_dataset(path_test)





import os
import cv2
import math
import torch
import numpy as np
import random

from torch.utils import data

class Dataset(data.Dataset):
    def __init__(
        self,
        data_path,
        split,
        image_folder,
        gt_folder,
        seed,
        do_transform,
    ):
        self.data_path = data_path
        self.split = split
        self.image_folder = image_folder
        self.gt_folder = gt_folder
        self.catesian2polar_height = 1250//2
        self.catesian2polar_width = math.ceil(1250*np.pi/16)*16
        self.n_classes = 4
        self.do_transform = do_transform

        self.files = os.listdir(f"{data_path}/{split}/{self.image_folder}")

        random.seed(seed)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        # Read image and mask
        img_name = self.files[index]
        img_path = f"{self.data_path}/{self.split}/{self.image_folder}/{img_name}"
        gt_path = f"{self.data_path}/{self.split}/{self.gt_folder}/{img_name}"

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        # Change gt
        gt[gt==11] = 3
        gt[gt==9] = 2

        # Preprocessing
        img, gt = self.preprocessing(img, gt)

        # Transform
        if self.do_transform:
            img, gt= self.transform(img, gt)

        # Normalize
        img = (img - img.mean()) / img.std()

        # HW -> CHW
        img = np.expand_dims(img, axis=0)

        # To tensor
        img = torch.from_numpy(img.copy()).float()
        gt = torch.from_numpy(gt.copy()).long()

        return img, gt

    def preprocessing(self, img, gt):
        img = cv2.warpPolar(
            img,
            (self.catesian2polar_height, self.catesian2polar_width),
            (1250/2, 1250/2),
            1250/2,
            cv2.INTER_CUBIC + cv2.WARP_POLAR_LINEAR
        )
        img = img[:,-336::]
        img = img.T
        img = img[::-1,:]

        gt = cv2.warpPolar(
            gt,
            (self.catesian2polar_height, self.catesian2polar_width),
            (1250/2, 1250/2),
            1250/2,
            cv2.INTER_NEAREST + cv2.WARP_POLAR_LINEAR
        )
        gt = gt[:,-336::]
        gt = gt.T
        gt = gt[::-1,:]

        return img, gt

    def transform(self, img, gt):
        # Horizontal flip
        flip = np.random.rand(1)[0]
        if flip >= 0.5:
            img = cv2.flip(img, 1)
            gt = cv2.flip(gt, 1)

        # Cut the image in middle and swap left and right parts
        pivot = math.floor(random.random() * (img.shape[1] + 1))

        left = img[:,0:pivot]
        right = img[:,pivot:img.shape[1]]
        img = np.concatenate((left, right), axis=1)

        left = gt[:,0:pivot]
        right = gt[:,pivot:gt.shape[1]]
        gt = np.concatenate((left, right), axis=1)

        return img, gt

if __name__ == "__main__":
    pass



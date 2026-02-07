import os
# import re
import sys
# import cv2

import numpy as np
# import SimpleITK as sitk

import torch
import torch.utils.data as data

from PIL import Image

import __init__
from  baseUtils.kit      import gaussianHeatmap
from  data_aug.transform import transformer, transformer4seg2d

class landmarks(data.Dataset):

    def __init__(self, prefix, phase, is_transform=True, sigma=10, num_landmark=19, size=[640, 800], use_background_channel=False, is_heatmap=True):

        self.is_heatmap = is_heatmap
        self.size = tuple(size)
        self.num_landmark = num_landmark
        self.use_background_channel = use_background_channel

        self.path_images = os.path.join(prefix, 'images')
        self.path_labels = os.path.join(prefix, 'labels')

        ##image and label file
        images = [os.path.join(self.path_images,i) for i in sorted(os.listdir(self.path_images))]
        labels = [os.path.join(self.path_labels,i) for i in sorted(os.listdir(self.path_labels))]
        if len(images) != len(labels):
            print('Images: ', len(images))
            print('Labels: ', len(labels))
            print('Image and label may be not match!')
            sys.exit(0)

        total_n = len(images)
        train_ratio, valid_ratio, test_ratio = 0.75, 0.25, 0.2
        train_n, valid_n = int(total_n*train_ratio), int(total_n*valid_ratio)

        if phase   == 'train':
            self.images = images[:train_n]
            self.labels = labels[:train_n]
        elif phase == 'validate':
            self.images = images[train_n:(train_n+valid_n)]
            self.labels = labels[train_n:(train_n+valid_n)]
        elif phase == 'test':
            self.images = images[(train_n+valid_n):]
            self.labels = labels[(train_n+valid_n):]
        else:
            raise Exception("Unknown phase: {phase}".fomrat(phase=phase))
        ##onif
        
        self.is_transform  = is_transform
        self.genHeatmap    = gaussianHeatmap(sigma, dim=len(size))
    ##ondef

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        
        ##load image
        img_arr    = self.readImage(self.images[index])
        
        ##load label
        point_list = self.readLandmark(self.labels[index])
        
        heatmap_list = [self.genHeatmap(point, self.size) for point in point_list]
        heatmap_arr  = np.array(heatmap_list)
        gt_arr  = heatmap_arr.astype(np.float32)

        # ## transform
        # if self.is_transform:
        #     img_arr, gt_arr = transformer(img_arr, gt_arr)

        img_arr  = img_arr.astype(np.float32)           ##conveting to float

        ret  = {'name': self.images[index]}
        ret['input'] = torch.from_numpy(img_arr.copy())
        ret['gt']    = torch.from_numpy(gt_arr.copy())

        return ret

    def readImage(self, path_image):
        '''Read image from path and return a numpy.ndarray in shape of c*w*h
        '''
        img      = Image.open(path_image)             ##RGB

        if os.path.splitext(path_image)[-1] == ".png":
            img = img.convert("RGB")
        ##onif

        self.origin_size = img.size  
        img = img.resize(self.size)               ##resize image

        img_arr  = np.array(img)                      ##PIL image to array
        
        # img_arr  = img_arr[:, :, :]                 ##height x width, because all channels are the same
        img_arr  = np.transpose(img_arr, (2, 1, 0))   ##height x width >>>> width x height
        # #########################################################################
        # img_pyd1 = cv2.pyrDown(img_arr)
        # img_pyd1 = np.pad(img_pyd1, ((0, (img_arr.shape[0] - img_pyd1.shape[0])), (0, (img_arr.shape[1] - img_pyd1.shape[1]))), "constant", constant_values=(0, 0))

        # img_pyd2 = cv2.pyrDown(img_pyd1)
        # img_pyd2 = np.pad(img_pyd2, ((0, (img_arr.shape[0] - img_pyd2.shape[0])), (0, (img_arr.shape[1] - img_pyd2.shape[1]))), "constant", constant_values=(0, 0))

        # img_arr  = np.stack((img_arr, img_pyd1, img_pyd2), axis=0) 
        # #########################################################################
        # img_arr  = np.expand_dims(img_arr, 0)         ##width x height >>>> channel x width x height
        # img_arr  = np.transpose(img_arr, (2, 1, 0))   ##height x width x channel >>>> channel x width x height

        # img_arr  = img_arr.astype(np.float32)           ##conveting to float

        return img_arr
    ##ondef

    def readLandmark(self, path_label):
        '''Read landmarks label from text file and return a list
        '''
        points = []
        with open(path_label) as f1:
            for i in range(self.num_landmark):
                landmark = f1.readline().rstrip('\n').split(',')
                # print(landmark)
                landmark = [int(i) for i in landmark]
                landmark = tuple(round(p*new/old) for p, new, old in zip(landmark, self.size, self.origin_size))
                points.append(landmark)
            ##onfor
        return points
    ##ondef

    def standardization(self, img_arr):
        mu       = np.mean(img_arr)
        sigma    = np.std(img_arr)
        img_norm = (img_arr - mu) / (sigma + 1e-20) # 加一个特小数防止除数为0
        return img_norm
    
    def normalize(self, img_arr):
        max_num  = np.max(img_arr)
        min_num  = np.min(img_arr)
        img_norm = (img_arr - min_num) / (max_num - min_num + 1e-20) # 加一个特小数防止除数为0
        return img_norm
##onclass


class LLimbDataset(data.Dataset):

    def __init__(self, dataset_directory, train=True, transform=True, size=[512, 512]):
        super(LLimbDataset, self).__init__()
        ##parameter
        self.train     = train
        self.transform = transform
        self.size      = tuple(size) ##image size

        ##image and label file
        self.path_images = os.path.join(dataset_directory, 'images')
        images = [os.path.join(self.path_images, i) for i in sorted(os.listdir(self.path_images))]

        self.path_labels = os.path.join(dataset_directory, 'labels')
        labels = [os.path.join(self.path_labels, i) for i in sorted(os.listdir(self.path_labels))]

        if len(images) != len(labels):
            print('Image and label may be not match!')
            sys.exit(0)
        ##onif

        ##
        total_n = len(images)
        train_ratio, valid_ratio = 0.8, 0.2
        train_n, valid_n         = int(total_n*train_ratio), int(total_n*valid_ratio)

        if train:
            self.images = images[:train_n]
            self.labels = labels[:train_n]
        else:
            # self.images = images[train_n:(train_n+valid_n)]
            # self.labels = labels[train_n:(train_n+valid_n)]
            self.images = images[train_n:]
            self.labels = labels[train_n:]
        ##onif
    ##ondef

    def __len__(self):
        return len(self.images)
    ##ondef

    def __getitem__(self, item):
        
        ##load image
        img_arr  = self.readImage(self.images[item])
        
        ##load label
        seg_arr  = self.readImage(self.labels[item], is_seg=True)

        ##augmentation
        if self.transform:
            img_arr, seg_arr = transformer4seg2d(img_arr=img_arr, seg_arr=seg_arr)
        ##onif

        ret  = {'name': self.images[item]}
        ret['data'] = torch.from_numpy(img_arr.copy())
        ret['seg']    = torch.from_numpy(seg_arr.copy())

        return ret

    def readImage(self, path_image, is_seg=False):
        '''Read image from path and return a numpy.ndarray in shape of y*x
        '''

        img      = Image.open(path_image)             ##RGB
        self.origin_size = img.size                   ##get image size
        img = img.resize(self.size)                   ##resize image

        img_arr  = np.array(img)                      ##PIL image to array
        # print("shape1:", img_arr.shape)
        img_arr  = np.transpose(img_arr, (2, 1, 0))   ##height x width >>>> width x height


        if not is_seg:
            img_arr  = img_arr.astype(np.float32)           ##conveting to float
        else:
            img_arr[img_arr!=0] = 1
            img_arr  = img_arr[0]
            img_arr  = img_arr.astype(np.int32)
        ##onif           

        return img_arr
    #ondef
##onclass
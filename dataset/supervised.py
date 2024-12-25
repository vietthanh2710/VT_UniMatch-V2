from dataset.transform import *
# from transform import *

from copy import deepcopy
import math
import numpy as np
import os
import random

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class SemiDataset(Dataset):
    def __init__(self, name, root, mode, size=None, id_path=None, nsample=None):
        """
        Args:
            name (str): Dataset name.
            root (str): Root directory of the dataset.
                /u03/thanhnv/2_DocumentSegmentation
            mode (str): Mode of operation ('train_l', 'train_u', or 'val').
            size (tuple): Target size of images (optional).
            id_path (str): Path to the file listing image paths.
            nsample (int): Number of samples for 'train_l'.
        """
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size

        # Determine the file path for train/validation IDs
        if id_path is None:
            id_path = os.path.join(root, 'train.txt' if mode in ['train_l', 'train_u'] else 'val.txt')

        # Read image paths from the ID file
        with open(id_path, 'r') as f:
            self.ids = f.read().splitlines()

        # Handle sampling for labeled training data
        if mode == 'train_l' and nsample is not None and nsample > len(self.ids):
            self.ids *= math.ceil(nsample / len(self.ids))
            self.ids = self.ids[:nsample]

    def __getitem__(self, item):
        # Image path from the IDs list
        id = self.ids[item]
        img_path = os.path.join(self.root, id)
        img = Image.open(img_path).convert('RGB')
        
        if self.mode == 'train_u':
            mask = Image.fromarray(np.zeros((img.size[1], img.size[0]), dtype=np.uint8))
        else:
            # Derive the corresponding mask path
            mask_path = img_path.replace('/images/', '/masks/').rsplit('.', 1)[0] + '.png'
            mask = Image.fromarray(np.array(Image.open(mask_path))) 
        
        if self.mode == 'val':
            img, mask = normalize(img, mask)
            return img, mask, id
        
        img, mask = resize(img, mask, (0.5, 2.0))
        ignore_value = 254 if self.mode == 'train_u' else 255


        img, mask = crop(img, mask, self.size, ignore_value)
        img, mask = hflip(img, mask, p=0.5)

        if self.mode == 'train_l':
            return normalize(img, mask)
        
        img_w, img_s1, img_s2 = deepcopy(img), deepcopy(img), deepcopy(img)

        if random.random() < 0.8:
            img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)
        img_s1 = transforms.RandomGrayscale(p=0.2)(img_s1)
        img_s1 = blur(img_s1, p=0.5)
        cutmix_box1 = obtain_cutmix_box(img_s1.size[0], p=0.5)

        if random.random() < 0.8:
            img_s2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s2)
        img_s2 = transforms.RandomGrayscale(p=0.2)(img_s2)
        img_s2 = blur(img_s2, p=0.5)
        cutmix_box2 = obtain_cutmix_box(img_s2.size[0], p=0.5)

        ignore_mask = Image.fromarray(np.zeros((mask.size[1], mask.size[0])))

        img_s1, ignore_mask = normalize(img_s1, ignore_mask)
        img_s2 = normalize(img_s2)

        mask = torch.from_numpy(np.array(mask)).long()
        ignore_mask[mask == 254] = 255

        return normalize(img_w), img_s1, img_s2, ignore_mask, cutmix_box1, cutmix_box2

    def __len__(self):
        return len(self.ids)
    

# dataset = SemiDataset(
#     name="DocumentSegmentation",
#     root="/u03/thanhnv/2_DocumentSegmentation",
#     mode="train_l",
#     size=640,  # Resize to 256x256
# )

# test_img = dataset[10][0]
# test_msk1 = dataset[10][1]
# test_msk2 = dataset[1][1].unsqueeze(dim=0)/255
# test_msk2 = test_msk2.long()
# x = torch.rand(size = (1,1,640,640))
# import torch.nn as nn
# import torch
# cri = nn.CrossEntropyLoss()
# print(x.shape,test_msk2.shape)
# loss = cri(x,test_msk2)
# print(loss)

# # from torchvision.utils import save_image
# # save_image(test_img, "/u03/thanhnv/2_DocumentSegmentation/test_img.png")
# # save_image(test_msk, "/u03/thanhnv/2_DocumentSegmentation/test_msk.png")

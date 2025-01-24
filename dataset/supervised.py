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

        if self.mode == 'infer':
            img = normalize(img)
            return img, id
        
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
            # img, mask = normalize(img, mask)
            # return img, mask, id # weighted fully
            return normalize(img, mask) # semi
        
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
    

CATEGORY_TO_IDX = {
    'bic_pru': 0,
    'card': 1,
    'diw': 2,
    'invoice': 3,
    'passport': 4,
    'tam_tru': 5
}

def get_category_idx_multi(paths):
    """
    Extract category index (0-5) from a list of image paths.
    
    Args:
        paths (list of str): List of image paths.
    
    Returns:
        list of int: List of category indices (0-5) extracted from paths.
                     Returns -1 for paths with categories not found in CATEGORY_TO_IDX.
    """
    results = []
    for path in paths:
        parts = path.split('/')
        try:
            img_idx = parts.index('images')
            category = parts[img_idx + 1]
            results.append(CATEGORY_TO_IDX.get(category, -1))  # Append -1 if category not found
        except (ValueError, IndexError):
            results.append(-1)
    return results

class SemiDataset_Weight(Dataset):
    def __init__(self, name, root, mode, size=None, id_path=None, nsample=None, weight = None):
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
            id_path = os.path.join(root, 'train.txt' if mode in ['train_l'] else 'val.txt')

        # Read image paths from the ID file
        with open(id_path, 'r') as f:
            self.ids = f.read().splitlines()
        self.folder_sample = get_category_idx_multi(self.ids)
        self.sample_weight = [weight[sample] for sample in self.folder_sample]
        # Handle sampling for labeled training data
        if mode == 'train_l' and nsample is not None and nsample > len(self.ids):
            self.ids *= math.ceil(nsample / len(self.ids))
            self.ids = self.ids[:nsample]


    def __getitem__(self, item):
        # Image path from the IDs list
        id = self.ids[item]
        img_path = os.path.join(self.root, id)
        img = Image.open(img_path).convert('RGB')
        
        mask_path = img_path.replace('/images/', '/masks/').rsplit('.', 1)[0] + '.png'
        mask = Image.fromarray(np.array(Image.open(mask_path))) 
        
        if self.mode == 'val':
            img, mask = normalize(img, mask)
            return img, mask, id
        
        img, mask = resize(img, mask, (0.5, 2.0))
        ignore_value = 255

        img, mask = crop(img, mask, self.size, ignore_value)
        img, mask = hflip(img, mask, p=0.5)

        if self.mode == 'train_l':
            img, mask = normalize(img, mask)
            return img, mask

    def __len__(self):
        return len(self.ids)
    

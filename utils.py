import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np

class CapsuleDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = []
        self.transform = transform
        
        for file in os.listdir(root_dir):
            if file.endswith('.png'):
                self.image_paths.append(os.path.join(root_dir, file))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        img = cv2.resize(img, (128, 128))
        img = img / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = torch.tensor(img, dtype=torch.float32)
        return img
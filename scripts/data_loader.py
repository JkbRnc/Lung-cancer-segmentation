import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

""" Data pipeline """
class BjorkeDataset(Dataset):
    def __init__(self, images, masks) -> None:
        self.images = images
        self.masks = masks
        self.n_samples = len(images)

    def __getitem__(self, index):
        """ Read image """
        image = cv2.imread(self.images[index], cv2.IMREAD_COLOR)
        image = image / 255.0  # Normalize
        image = np.transpose(image, (2, 0, 1))  # Channel first
        image = image.astype(np.float32)
        image = torch.from_numpy(image)

        """ Read mask """
        mask = cv2.imread(self.masks[index], cv2.IMREAD_GRAYSCALE)
        mask = mask / 255.0
        mask = np.expand_dims(mask, axis=0)  # Add channel
        mask = mask.astype(np.float32)
        mask = torch.from_numpy(mask)

        return image, mask

    def __len__(self):
        return self.n_samples

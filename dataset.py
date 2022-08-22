import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import random
import torch

class BatchData(Dataset):
    def __init__(self, images, labels, input_transform=None, is_ori = True):
        self.images = images
        self.labels = labels
        self.is_ori = is_ori
        self.imgs = (self.images,self.labels)
        self.input_transform = input_transform

    def __getitem__(self, index):
        image = self.images[index]
        if self.is_ori:
            image = Image.fromarray(np.uint8(image))
        label = self.labels[index]
        if self.input_transform is not None:
            image = self.input_transform(image)
        label = torch.LongTensor([label])
        return image, label
    def get_labels(self):
        return self.labels
    def __len__(self):
        return len(self.images)

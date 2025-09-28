import copy
import datetime
import io
import json
import os
import pandas as pd
from PIL import Image
import time
import torch
import torchvision
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from transformers import ViTForImageClassification, AutoImageProcessor
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

class CVDataset(Dataset):
    def __init__(self, images, labels, processor=None):
        self.images = images
        self.labels = labels
        self.processor = processor

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_data = self.images[idx]
        label = self.labels[idx]

        img = Image.open(io.BytesIO(img_data)).convert('RGB')
        processed = self.processor(images=img, return_tensors="pt")
        processed_img = processed['pixel_values'].squeeze(0)

        return processed_img, label

        
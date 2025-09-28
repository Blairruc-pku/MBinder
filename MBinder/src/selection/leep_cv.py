from datasets import load_dataset

from transformers import AutoImageProcessor
from transformers import AutoModelForImageClassification
import torch
from torch import nn
import numpy as np
import math

import argparse
import os

import datetime

start_time = datetime.datetime.now()

PATH = os.path.dirname(__file__)
ROOT_PATH = os.path.dirname(os.path.dirname(__file__))

def leep_cv(model, image_processor, dataset, label_num):
    """
    Compare to not_consider_theta version
    Instead of a simple output of max possible label
    Inference output becomes a distribution of label z
    Thus LEEP score considers the uncertainty when model faces ambiguous output
        e.g. Model A judges an image resulting in score (0.2,0.8) 
            while Model B resulting in score (0.45,0.55). Previous calculation considers
            both as (0,1), but obviously, Model A gives a more confident result.
    """
    # step 1 get the label z through inference
    label_z = []
    # label_y is the original label
    label_y = []
    for data in dataset:
        inputs = image_processor(data['image'], return_tensors="pt").to("cuda")
        with torch.no_grad():
            logits = model(**inputs).logits
        theta_tensor = nn.functional.softmax(logits, dim=-1).cpu() # need to copy from gpu first
        theta_distribution = theta_tensor.numpy()[0]
        #print(theta_distribution)
        label_z.append(theta_distribution)
        # label_z saves the theta distribution of each image that uses for LEEP calculation
        label_y.append(data['label'])
    print("Finish Inference")
    print(model.classifier.weight.shape)
    print(label_y[0])
    print(label_z[0])
    # step 2 get joint distribution P(y,z)
    # TODO:
    total_num = len(label_y)
    #y_size = len(dataset.features["label"].names)
    y_size = label_num
    z_size = len(label_z[0])
    joint_distribution = np.zeros((y_size, z_size))
    for i in range(len(label_y)):
        # for each image
        for j in range(z_size):
            # add expectation of each possible labels
            joint_distribution[label_y[i]][j] += label_z[i][j]
    joint_distribution = joint_distribution / total_num
    #print(joint_distribution)

    # step 3 get z distribution P(z)
    z_distribution = np.array(label_z)
    z_distribution = np.sum(z_distribution, axis=0)
    z_distribution = z_distribution / total_num
    #print(z_distribution)

    # step 4 get conditional distribution P(y|z)
    conditional_distribution = joint_distribution / z_distribution
    #print(conditional_distribution)

    # step 5 get LEEP score
    LEEP = 0
    for i in range(len(label_y)):
        LEEP_single_y = 0
        current_y_label = label_y[i]
        for j in range(z_size):
            LEEP_single_y += (conditional_distribution[current_y_label][j]*label_z[i][j])
        LEEP += (math.log(LEEP_single_y))
    LEEP = LEEP / total_num

    return LEEP


from datasets import load_dataset

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

import torch
from torch import nn

import numpy as np
import math

import argparse
import os
import datetime

start_time = datetime.datetime.now()

PATH = os.path.dirname(__file__)

"""
Similar to image_processor, text need a tokenizer to get the correct inputs
Different tasks need to tokenize different text
    e.g. MNLI inputs a sentence pair 'premise' and 'hypothesis'
        that both needs to tokenize.
"""
def leep_nlp(model, tokenizer, dataset, label_num):
    """
    Calculate LEEP score on NLP task/model.
    Model should be an AutoModelForSequenceClassification instance.
    Tokenizer should be the model's tokenizer.
    Dataset should be a list of dict in which contains 'text' and 'label'.
    """
    # step 1 get the label z through inference
    label_z = []
    # label_y is the original label
    label_y = []
    for data in dataset:
        inputs = tokenizer(data['text'], return_tensors="pt", truncation=True, padding=True, max_length=128).to("cuda")
        with torch.no_grad():
            logits = model(**inputs).logits
        theta_tensor = nn.functional.softmax(logits, dim=-1).cpu()  # need to copy from gpu first
        theta_distribution = theta_tensor.numpy()[0]
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

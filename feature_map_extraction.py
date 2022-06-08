# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 20:17:36 2022

@author: Yen
"""

from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
import requests
from torchvision import transforms, models
import matplotlib.pyplot as plt

def get_features_forResnet(model):
    l = []
    for key, value in model._modules.items():
        # cheeck recursive condition
        if len(value._modules) > 0:
            l = l + get_features_forResnet(value)
        else:
            if isinstance(value, nn.Conv2d):
                if 'conv' in key:
                    l.append(value)
    return l

def get_features_forVGG(model):
    l = []
    for key, value in model._modules.items():
        # cheeck recursive condition
        if len(value._modules) > 0:
            l = l + get_features_forVGG(value)
        else:
            if isinstance(value, nn.Conv2d):
                l.append(value)
    return l

class myModel(nn.Module):
    def __init__(self, net, l):
        super().__init__()
        self.net = net
        self.l = l
    def forward(self, x):
        return_list = []
        for key, value in self.net._modules.items():
            x = value(x)
            if key in self.l:
                return_list.append(x)
        return return_list
    
def plot_feature_map(output_features, layer):
    fig = plt.figure(figsize=(10,10))
    for i in range(0,4):
        plt.subplot(2,2,i+1)
        plt.imshow(output_features[layer][0,i,:,:].detach().numpy() , cmap='gray')
    plt.show()
    
def load_image(img_path, max_size=400, shape=None):
    ''' Load in and transform an image, making sure the image
       is <= 400 pixels in the x-y dims.'''
    if "http" in img_path:
        response = requests.get(img_path)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(img_path).convert('RGB')
    
    # large images will slow down processing
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)
    
    if shape is not None:
        size = shape
        
    in_transform = transforms.Compose([
                        transforms.Resize(size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), 
                                             (0.229, 0.224, 0.225))])

    # discard the transparent, alpha channel (that's the :3) and add the batch dimension
    image = in_transform(image)[:3,:,:].unsqueeze(0)
    
    return image

def get_feature_from_VGG19(image_path):
    # hyperparameter (tunable)
    layer_selection = ['2','5','8','11','15']
    # select which layer (index format)
    index = 0
    
    content = load_image(image_path)
    vgg_19 = models.vgg19(pretrained=True)
    l = get_features_forVGG(vgg_19)
    conv_model = nn.Sequential(*l)
    myNet = myModel(conv_model, layer_selection)
    output = myNet(content)
    plot_feature_map(output, index)
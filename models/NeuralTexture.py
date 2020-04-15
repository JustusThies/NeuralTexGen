import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from .base_model import BaseModel
from . import networks
import numpy as np
import functools
from PIL import Image
from util import util
from torchvision import models
from collections import namedtuple

################
###  HELPER  ###
################
INVALID_UV = -1.0

#####################################
########   static texture   #########
#####################################
class StaticNeuralTexture(nn.Module):
    def __init__(self, texture_dimensions, texture_features, random_init=False):
        super(StaticNeuralTexture, self).__init__()
        self.texture_dimensions = texture_dimensions #256 #texture dimensions
        self.out_ch = texture_features # output feature, after evaluating the texture
        if random_init:
            self.register_parameter('data', torch.nn.Parameter(torch.randn(1, self.out_ch, self.texture_dimensions, self.texture_dimensions, requires_grad=True)))
        else:
            self.register_parameter('data', torch.nn.Parameter(torch.zeros(1, self.out_ch, self.texture_dimensions, self.texture_dimensions, requires_grad=True)))
        ####

    def forward(self, uv_inputs):     
        b = uv_inputs.shape[0] # batchsize
        if b != 1:
            print('ERROR: NeuralTexture forward only implemented for batchsize==1')
            exit(-1)
        uvs = torch.stack([uv_inputs[:,0,:,:], uv_inputs[:,1,:,:]], 3)
        return torch.nn.functional.grid_sample(self.data, uvs, mode='bilinear', padding_mode='border')

    def regularizer(self):
        return 0.0

    def SaveToFile(self, filename):
        image_tensor = self.data[0,0:3,:,:]
        image_numpy = image_tensor.clone().cpu().float().detach().numpy()
        image_numpy = np.clip(image_numpy, -1.0, 1.0)
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
        image_numpy = image_numpy.astype(np.uint8)
        image_pil = Image.fromarray(image_numpy)
        image_pil.save(filename)

class HierarchicalStaticNeuralTexture(nn.Module):
    def __init__(self, texture_dimensions, texture_features, random_init=False):
        super(HierarchicalStaticNeuralTexture, self).__init__()
        self.texture_dimensions = texture_dimensions #256 #texture dimensions
        self.out_ch = texture_features # output feature, after evaluating the texture
        if random_init:
            self.register_parameter('data', torch.nn.Parameter(torch.randn(1, self.out_ch, 2 * self.texture_dimensions, self.texture_dimensions, requires_grad=True)))
        else:
            self.register_parameter('data', torch.nn.Parameter(torch.zeros(1, self.out_ch, 2 * self.texture_dimensions, self.texture_dimensions, requires_grad=True)))
        ####

    def forward(self, uv_inputs):
        b = uv_inputs.shape[0] # batchsize
        if b != 1:
            print('ERROR: HierarchicalStaticNeuralTexture forward only implemented for batchsize==1')
            exit(-1)
        uvs = torch.stack([uv_inputs[:,0,:,:], uv_inputs[:,1,:,:]], 3)

        # hard coded pyramid
        texture_id=0
        offsetY = 0
        w = self.texture_dimensions
        self.high_level_tex = self.data[texture_id:texture_id+1, :, offsetY:offsetY+w, :w]
        high_level = torch.nn.functional.grid_sample(self.high_level_tex, uvs, mode='bilinear', padding_mode='border')
        offsetY += w
        w = w // 2
        self.medium_level_tex = self.data[texture_id:texture_id+1, :, offsetY:offsetY+w, :w]
        medium_level = torch.nn.functional.grid_sample(self.medium_level_tex, uvs, mode='bilinear', padding_mode='border')
        offsetY += w
        w = w // 2
        self.low_level_tex = self.data[texture_id:texture_id+1, :, offsetY:offsetY+w, :w]
        low_level = torch.nn.functional.grid_sample(self.low_level_tex, uvs, mode='bilinear', padding_mode='border')
        offsetY += w
        w = w // 2
        self.lowest_level_tex = self.data[texture_id:texture_id+1, :, offsetY:offsetY+w, :w]
        lowest_level = torch.nn.functional.grid_sample(self.lowest_level_tex, uvs, mode='bilinear', padding_mode='border')

        return high_level + medium_level + low_level + lowest_level

    def regularizer(self, high_weight=8.0, medium_weight=2.0, low_weight=1.0, lowest_weight=0.0):
        regularizerTex  = torch.mean(torch.pow( self.high_level_tex,   2.0 )) * high_weight
        regularizerTex += torch.mean(torch.pow( self.medium_level_tex, 2.0 )) * medium_weight
        regularizerTex += torch.mean(torch.pow( self.low_level_tex,    2.0 )) * low_weight
        regularizerTex += torch.mean(torch.pow( self.lowest_level_tex, 2.0 )) * lowest_weight
        return regularizerTex



    def SaveToFile(self, filename):
        dim_range = torch.arange(0, self.texture_dimensions, dtype=torch.float) / (self.texture_dimensions - 1.0) * 2.0 - 1.0
        ones = torch.ones(self.texture_dimensions, dtype=torch.float) * 1.0
        v = torch.ger(dim_range, ones) # outer product
        u = torch.ger(ones, dim_range)
        uv_id = torch.cat([u.unsqueeze(0).unsqueeze(0),v.unsqueeze(0).unsqueeze(0)], 1)
        uv_id = uv_id.to(self.data.device)

        image_tensor = self.forward(uv_id)[0,0:3,:,:]
        image_numpy = image_tensor.clone().cpu().float().detach().numpy()
        image_numpy = np.clip(image_numpy, -1.0, 1.0)
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
        image_numpy = image_numpy.astype(np.uint8)
        image_pil = Image.fromarray(image_numpy)
        image_pil.save(filename)



###################################
########   simple array   #########
###################################

class NeuralArray(nn.Module):
    def __init__(self, dim, random_init=False):
        super(NeuralArray, self).__init__()
        self.dim = dim
        if random_init:
            self.register_parameter('data', torch.nn.Parameter(torch.randn(self.dim, requires_grad=True)))
        else:
            self.register_parameter('data', torch.nn.Parameter(torch.zeros(self.dim, requires_grad=True)))
        ####

    def forward(self, id):
        return self.data[id]

    def regularizer_zero(self):
        return torch.mean(torch.pow( self.data, 2.0 ))

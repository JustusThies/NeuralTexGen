import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from .base_model import BaseModel
from . import networks
import numpy as np
import functools
from util import util
from torchvision import models
from collections import namedtuple


from . import VGG_LOSS
from . import NeuralTexture

################
###  HELPER  ###
################
INVALID_UV = -1.0


def define_Weights(opt, gpu_ids=[]):
    net = None
    net = NeuralTexture.NeuralArray(dim=opt.dataset_size*3, random_init=False) # init with zeros
    return networks.init_net(net, opt.init_type, opt.init_gain, gpu_ids)

def define_Texture(opt, gpu_ids=[]):
    net = None
    random_init=False
    #net = NeuralTexture.StaticNeuralTexture(texture_dimensions=opt.tex_dim, texture_features=3, random_init=random_init)   
    net = NeuralTexture.HierarchicalStaticNeuralTexture(texture_dimensions=opt.tex_dim, texture_features=3, random_init=random_init)   
    return networks.init_net(net, opt.init_type, opt.init_gain, gpu_ids)


class RGBTexturesModel(BaseModel):
    def name(self):
        return 'RGBTexturesModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['G_total', 'G_L1', 'G_L1_Diff', 'G_VGG', 'G_TexReg']

        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['input_uv', 'fake', 'target']
        self.model_names = ['texture']

        # load/define networks
        self.texture = define_Texture(opt,  self.gpu_ids)

        # optimizer
        if self.isTrain:
            # define loss functions
            self.criterionL1 = torch.nn.L1Loss(reduction='mean')
            self.criterionL1Smooth = torch.nn.SmoothL1Loss(reduction='mean')
            self.criterionL2 = torch.nn.MSELoss(reduction='mean')

            if self.opt.lambda_VGG > 0.0:
                self.vggloss = VGG_LOSS.VGGLOSS().to(self.device)

            # initialize optimizers
            self.optimizers = []
            self.optimizer_T = torch.optim.Adam(self.texture.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            #self.optimizer_T = torch.optim.SGD(self.texture.parameters(), lr=opt.lr, momentum=0)
            self.optimizers.append(self.optimizer_T)

    def WriteTextureToFile(self, filename):
        self.texture.SaveToFile(filename)

    def set_input(self, input):
        self.target = input['TARGET'].to(self.device)
        self.input_uv = input['UV'].to(self.device)             
        self.image_paths = input['paths']

        self.input_id = input['internal_id']

    def forward(self):
        # loop over batch elements
        batch_size = self.target.shape[0]

        self.fake = []

        for b in range(0,batch_size):
            fake =  self.texture(self.input_uv[b:b+1])
            self.fake.append(fake)

        self.fake = torch.cat(self.fake, dim=0)


    def backward_G(self, epoch):

        mask = ( (self.input_uv[:,0:1,:,:] != INVALID_UV) | (self.input_uv[:,1:2,:,:] != INVALID_UV) )
        eps = 0.0001
        sum_mask = torch.sum(mask) + eps
        #if sum_mask == 0:
        #    return
        num_pixels_max = mask.shape[-2] * mask.shape[-1]
        mask_weight = (num_pixels_max) / sum_mask
        mask = torch.cat([mask,mask,mask], 1)
        def masked(img):
            return torch.where(mask, img, torch.zeros_like(img))


        ## absolute rendering error
        self.loss_G_L1_Rendering = 0.0
        if self.opt.lambda_L1>0.0:
            self.loss_G_L1_Rendering += self.opt.lambda_L1 * self.criterionL1(masked(self.fake), masked(self.target) ) / mask_weight


        ## loss based on image differences (more invariant to color shifts)
        self.loss_G_L1_Diff = 0.0
        if self.opt.lambda_L1_Diff>0.0:
            fake_diff_x = self.fake[:,:,:,1:] - self.fake[:,:,:,:-1]
            fake_diff_y = self.fake[:,:,1:,:] - self.fake[:,:,:-1,:]
            target_masked = masked(self.target)
            target_diff_x = target_masked[:,:,:,1:] - target_masked[:,:,:,:-1]
            target_diff_y = target_masked[:,:,1:,:] - target_masked[:,:,:-1,:]
            self.loss_G_L1_Diff += self.opt.lambda_L1_Diff * self.criterionL1(fake_diff_x, target_diff_x) / mask_weight
            self.loss_G_L1_Diff += self.opt.lambda_L1_Diff * self.criterionL1(fake_diff_y, target_diff_y) / mask_weight


        ## VGG loss
        self.loss_G_VGG_Rendering = 0.0
        if self.opt.lambda_VGG>0.0:
            self.loss_G_VGG_Rendering += self.opt.lambda_VGG * self.vggloss(masked(self.fake), masked(self.target))


        ## texture regularizer
        if self.opt.lambda_Reg_Tex>0.0:
            self.loss_G_TexReg = self.opt.lambda_Reg_Tex * self.texture.regularizer()

        self.loss_G_total = self.loss_G_L1_Rendering + self.loss_G_VGG_Rendering + self.loss_G_TexReg + self.loss_G_L1_Diff

        self.loss_G_total.backward()

    def reset_gradients(self):
        self.optimizer_T.zero_grad()

    def compute_gradients(self, epoch, epoch_iter):
        self.forward()
        self.backward_G(epoch)

    def step_gradients(self):
        self.optimizer_T.step()


    def optimize_parameters(self, epoch, epoch_iter):
        self.forward()
        self.optimizer_T.zero_grad()
        self.backward_G(epoch)
        self.optimizer_T.step()


import os.path
import random
import torchvision.transforms as transforms
import torch
import numpy as np
from data.base_dataset import BaseDataset
from PIL import Image
from util import util
from scipy.misc import imresize

def make_dataset_exr_ids(dir):
    ids = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if any(fname.endswith(extension) for extension in ['.exr', '.EXR']):
                id_str = fname[:-4]
                i = int(id_str)
                ids.append(i)
    ids = sorted(ids)
    return ids


class UVDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt

        # directories
        self.dataroot = opt.dataroot
        self.image_dir = os.path.join(opt.dataroot, 'images')
        self.uvs_dir = os.path.join(opt.dataroot, 'uvs')

        # debug print
        if opt.verbose:
            print('load sequence:', self.dataroot)
            print('\timage_dir:', self.image_dir)
            print('\tuvs_dir:', self.uvs_dir)

        # generate index maps
        self.uvs_ids = make_dataset_exr_ids(self.uvs_dir)
        self.n_frames_total = len(self.uvs_ids)

        if opt.verbose:
            print('\tnum frames:', self.n_frames_total)
        

    def getSampleWeights(self):
        weights = np.ones((self.n_frames_total))
        return weights

    def __getitem__(self, global_index):
        # select frame from sequence
        index = global_index
        id = self.uvs_ids[index]

        ## UV
        uv_fname = os.path.join(self.uvs_dir, str(id) + '.exr')
        uv_numpy = util.load_exr(uv_fname)

        UV = transforms.ToTensor()(uv_numpy.astype(np.float32))
        UV = torch.where(UV > 1.0, torch.zeros_like(UV), UV)
        UV = torch.where(UV < 0.0, torch.zeros_like(UV), UV)
        UV = 2.0 * UV - 1.0

        ## img
        img_fname = os.path.join(self.image_dir, str(id) + '.jpg')
        img_numpy = np.asarray(Image.open(img_fname))
        img_numpy = imresize(img_numpy, (UV.shape[1],UV.shape[2]), interp='bicubic') # <<< img resize

        TARGET = transforms.ToTensor()(img_numpy.astype(np.float32))/255.0
        #TARGET = torch.pow(TARGET, 1.0/2.2) # gamma correction
        TARGET = 2.0 * TARGET - 1.0

        #################################
        weight = 1.0 / self.n_frames_total
        
        return {'TARGET': TARGET, 'UV': UV,
                'paths': img_fname,
                'internal_id': index,
                'weight': np.array([weight]).astype(np.float32)}

    def __len__(self):
        return self.n_frames_total

    def name(self):
        return 'UVDataset'

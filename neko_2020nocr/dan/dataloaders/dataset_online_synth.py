# coding:utf-8
import torch
from torch.utils.data import Dataset
from neko_2020nocr.dan.dataloaders.dataset_common import keepratio_resize;
from neko_sdk.ocr_modules.trdg_driver.corpus_data_generator_driver import neko_random_string_generator;
from neko_sdk.ocr_modules.trdg_driver.corpus_data_generator_driver import neko_skip_missing_string_generator;
import numpy as np
import cv2
import random;


class nekoOLSDataset(Dataset):

    def load_random_generator(this,root,maxT):
        meta=torch.load(root);
        g = neko_random_string_generator(meta, meta["bgims"],max_len=maxT);
        this.nSamples = 19999999;

        return g;

    def __init__(this, root=None, ratio=None, img_height = 32, img_width = 128,
        transform=None, global_state='Test',maxT=25):
        this.generator = this.load_random_generator(root)
        # Admit it=== you can never exhaust this. But let's at least set a number for an epoch
        this.global_state = global_state
        this.load_random_generator(root);
        this.transform = transform
        this.img_height = img_height
        this.img_width = img_width
        # Issue12
        this.target_ratio = img_width / float(img_height)


    def __len__(this):
        return this.nSamples

    def __getitem__(this, index):
        img,label=this.generator.random_clip()
        try:
            img = keepratio_resize(img,this.img_height,this.img_width,this.target_ratio,True);
        except:
            print('Size error for %d' % index)
            return this[index + 1]
        img = img[:, :, np.newaxis]
        if this.transform:
            img = this.transform(img)

        sample = {'image': img, 'label': label}
        return sample

class nekoOLSCDataset(nekoOLSDataset):
    def load_random_generator(this,root):
        meta=torch.load(root);
        g = neko_skip_missing_string_generator(meta, meta["bgims"]);
        this.nSamples = g.nSamples;
        return g;

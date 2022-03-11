# coding:utf-8
import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import sampler
import torchvision.transforms as transforms
import lmdb
import six
import sys
from PIL import Image
import numpy as np
from neko_2020nocr.dan.dataloaders.dataset_common import keepratio_resize;
import pdb
import os
import torchvision
import cv2
# ms : multi_sources
# semi: Semi supervision
class lmdbDataset_single_labeled(Dataset):
    def init_etc(this):
        pass;
    def __init__(this, root, img_height = 32, img_width = 128,
        transform=None, global_state='Test',maxT=25):
        this.maxT=maxT;
        this.transform = transform
        this.img_height = img_height
        this.img_width = img_width
        this.global_state=global_state;
        # Issue12
        this.target_ratio = img_width / float(img_height)
        env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)
        if not env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)
        with env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()))
            this.nSamples = nSamples
        this.root=root;
        this.env=env;
        this.init_etc();
        pass;


    def __len__(this):
        return this.nSamples
    def __getitem__(this, index):
        if( index > len(this)):
            print('index range error');
            index = index%len(this)+1;
        if(index==0):
            print("0 base detected. adding one");
            index+=1;
        with this.env.begin(write=False) as txn:
            img_key = 'image-%09d' % index
            try:
                imgbuf = txn.get(img_key.encode())
                buf = six.BytesIO()
                buf.write(imgbuf)
                buf.seek(0)
                img = Image.open(buf)
            except:
                print('Corrupted image for %d' % index)
                return this[index + 1]
            label_key = 'label-%09d' % index
            label = str(txn.get(label_key.encode()).decode('utf-8'));
            if len(label) > this.maxT-1 and this.global_state == 'Train':
                print('sample too long')
                return this[index + 1]
            try:
                img = keepratio_resize(img,this.img_height,this.img_width,this.target_ratio,True)
            except:
                print('Size error for %d' % index)
                return this[index + 1]
            img = img[:,:,np.newaxis]
            if this.transform:
                img = this.transform(img)
            sample = {'image': img, 'label': label}
            return sample

class lmdbDataset_single_unlabeled(Dataset):
    def init_etc(this):
        pass;
    def __init__(this, root, img_height = 32, img_width = 128,
        transform=None, global_state='Test',maxT=25):
        this.maxT=maxT;
        this.transform = transform
        this.img_height = img_height
        this.img_width = img_width
        this.global_state=global_state;
        # Issue12
        this.target_ratio = img_width / float(img_height)
        env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)
        if not env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)
        with env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()))
            this.nSamples = nSamples
        this.root=root;
        this.env=env;
        this.init_etc();
        pass;


    def __len__(this):
        return this.nSamples
    def __getitem__(this, index):
        if( index > len(this)):
            print('index range error');
            index = index%len(this)+1;
        if(index==0):
            print("0 base detected. adding one");
            index+=1;
        with this.env.begin(write=False) as txn:
            img_key = 'image-%09d' % index
            try:
                imgbuf = txn.get(img_key.encode())
                buf = six.BytesIO()
                buf.write(imgbuf)
                buf.seek(0)
                img = Image.open(buf)
            except:
                print('Corrupted image for %d' % index)
                return this[index + 1]
            label_key = 'label-%09d' % index
            label = str(txn.get(label_key.encode()).decode('utf-8'));
            if len(label) > this.maxT-1 and this.global_state == 'Train':
                print('sample too long')
                return this[index + 1]
            try:
                img = keepratio_resize(img,this.img_height,this.img_width,this.target_ratio,True)
            except:
                print('Size error for %d' % index)
                return this[index + 1]
            img = img[:,:,np.newaxis]
            if this.transform:
                img = this.transform(img)
            sample = {'image': img}
            return sample

class lmdbDataset_ms(Dataset):
    def init_etc(this):
        pass;
    def init_call(this,datasets, ratio=None, global_state='Test',repeat=1):
        this.repos=[]
        this.ratio=[];
        this.global_state = global_state
        this.repeat=repeat;
        this.nSamples = 0;
        this.maxlen=0;
        for i in range(0,len(datasets)):
            this.repos.append(datasets[i])
            length=len(this.repos[-1])
            if(length>this.maxlen):
                this.maxlen=length;
            this.nSamples+=length;
        if ratio != None:
            assert len(datasets) == len(datasets) ,'length of ratio must equal to length of roots!'
            for i in range(0,len(datasets)):
                this.ratio.append(ratio[i] / float(sum(ratio)))
        else:
            for i in range(0,len(datasets)):
                this.ratio.append(this.repos[i].nSamples / float(this.nSamples))

    def __fromwhich__(this ):
        rd = random.random()
        total = 0
        for i in range(0,len(this.ratio)):
            total += this.ratio[i]
            if rd <= total:
                return i

    def __len__(this):
        return this.nSamples

    def __getitem__(this, index):
        fromwhich = this.__fromwhich__()
        if this.global_state == 'Train':
            index = random.randint(1,this.maxlen)
        index = index % len(this.repos[fromwhich])
        assert index <= len(this), 'index range error'
        index += 1
        sample=this.repos[fromwhich][index]
        return sample

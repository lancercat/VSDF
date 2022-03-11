# coding:utf-8
from __future__ import print_function


#---------------------------------------------------------
#--------------------------Begin--------------------------
#---------------------------------------------------------
# Cmon'man, It's 2020 and we still need lexicon?
import cv2;
import torch;
import numpy as np
import pylcs;


def keepratio_resize(img, h, w,rgb=False):
    cur_ratio = img.shape[1] / float(img.shape[0])
    target_ratio = w / float(h)
    mask_height = h
    mask_width = w
    img = np.array(img)
    if rgb==False and len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif rgb==True and len(img.shape)==1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    if cur_ratio > target_ratio:
        cur_target_height = h
        # print("if", cur_ratio, self.target_ratio)
        cur_target_width = w
    else:
        cur_target_height = h
        # print("else",cur_ratio,self.target_ratio)
        cur_target_width = int(h * cur_ratio)
    img = cv2.resize(img, (cur_target_width, cur_target_height))
    start_x = int((mask_height - img.shape[0]) / 2)
    start_y = int((mask_width - img.shape[1]) / 2)
    mask = np.zeros([mask_height, mask_width,3]).astype(np.uint8)
    mask[start_x: start_x + img.shape[0], start_y: start_y + img.shape[1]] = img
    img = mask
    bmask=None;
    return img,bmask
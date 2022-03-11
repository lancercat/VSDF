import cv2;
import numpy as np;
def keepratio_resize(img,img_height,img_width,target_ratio,gray=True):
    cur_ratio = img.size[0] / float(img.size[1])

    mask_height = img_height
    mask_width = img_width
    img = np.array(img)
    if gray and len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if cur_ratio > target_ratio:
        cur_target_height = img_height
        # print("if", cur_ratio, this.target_ratio)
        cur_target_width = img_width
    else:
        cur_target_height = img_height
        # print("else",cur_ratio,this.target_ratio)
        cur_target_width = int(img_height * cur_ratio)
    img = cv2.resize(img, (cur_target_width, cur_target_height))
    start_x = int((mask_height - img.shape[0] ) /2)
    start_y = int((mask_width - img.shape[1] ) /2)
    mask = np.zeros([mask_height, mask_width]).astype(np.uint8)
    mask[start_x : start_x + img.shape[0], start_y : start_y + img.shape[1]] = img
    img = mask
    return img
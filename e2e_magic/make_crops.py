import os.path

import torch

from neko_sdk import libnorm
import numpy as np
import cv2
def handle_line(im,line):
    fields=line.split(",",10);
    cords=np.array([int(i) for i in fields[:8]]).reshape([4,2]);
    im=libnorm.norm(im,cords, None);
    return im

def handle_file(im,pred):
    im=cv2.imread(im);
    linims=[];
    with open(pred,"r") as fp:
        for l in fp:
            c=handle_line(im,l.strip());
            linims.append(c);
    return linims;
def handle_det(pred_root,img_root,crop_dst,predfmt="res_img_{0:0=5d}.txt",imgfmt="ts_img_{0:0=5d}.jpg",imgcnt=9000):
    c=0
    cdic={}
    for i in range(1,imgcnt+1):
        imname=imgfmt.format(i);
        predname=predfmt.format(i);
        cl=handle_file(os.path.join(img_root,imname),os.path.join(pred_root,predname));
        cdic[i] = [];
        for cr in cl:
            cv2.imwrite(os.path.join(crop_dst,str(c)+".jpg"),cr);
            cdic[i].append(c);
            c+=1;
    torch.save(cdic,os.path.join(crop_dst,"meta.pt"));

if __name__ == '__main__':
    handle_det("/run/media/lasercat/projects_001/mlt-test/detect_results/","/run/media/lasercat/projects_001/mlt-test/raw_images/","/run/media/lasercat/projects_001/mlt-test/crops/")
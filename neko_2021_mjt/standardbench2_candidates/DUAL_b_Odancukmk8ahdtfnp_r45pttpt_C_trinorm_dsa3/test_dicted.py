# coding:utf-8
from __future__ import print_function

from dicted_eval_configs import dan_mjst_dict_eval_cfg
from neko_2021_mjt.lanuch_std_test import testready
import os;

def get_tester():
    import sys
    if(len(sys.argv)<2):
        argv = ["Meeeeooooowwww",
                "/run/media/lasercat/ssddata/cvpr22_candidate/8ahdttpt/",
                "_E4",
                "/run/media/lasercat/ssddata/cvpr22_candidate/8ahdttpt/",
                ]
    else:
        argv=sys.argv;
    te_meta_path_mjst = os.path.join("/home/lasercat/ssddata/", "dicts", "dab62cased.pt");

    return testready(argv,dan_mjst_dict_eval_cfg,temeta=te_meta_path_mjst)

#---------------------------------------------------------
#--------------------------Begin--------------------------
#---------------------------------------------------------
# Cmon'man, It's 2021 and we still need lexicon?
import cv2;
import torch;
import numpy as np
import pylcs;
from neko_sdk.ocr_modules.img_eval import keepratio_resize

def img_test(img,runner,args):
    img=cv2.imread(img)
    imgr=keepratio_resize(img,32,128)/255.
    res=runner.test_im(torch.tensor(imgr).float().unsqueeze(0).unsqueeze(0).cuda(),args);
    return  res[0];
def dicted_test(img,runner,globalcache,lex):
    res=runner.test_img(0,img,
                globalcache);
    mind=9999;
    p=res;
    lns=[]
    with open(lex,"r") as fp:
        lns=[i.strip() for i in fp];
    lex=lns[0].split(",");
    iss=[];
    for i in lex:
        if(len(i)==0):
            continue;
        e=pylcs.edit_distance(res.lower(),i.lower())
        if(e<mind):
            mind=e;
            p=i
            iss.append(i)
    if(p==""):
        print("???")
    # if(p.lower()!=res.lower()):
    #     print(res,"->",p)
    return p
def rundictiiitm():
    runner,globalcache,mdict=get_tester();
    err = 0;
    for i in range(3000):

        with open("/home/lasercat/ssddata/iiit5kdicted/%d.gt" % i, "r") as fp:
            gts = [k.strip() for k in fp];
        try:
            res = dicted_test("/home/lasercat/ssddata/iiit5kdicted/%d.jpg" % i,
                              runner,
                              globalcache, "/run/media/lasercat/ssddata/iiit5kdicted/%d.lexm" % i);
            if (res != gts[0]):
                err += 1
                print(gts, "vs", res, "at", i);
        except:
            err += 1
    print(1 - err / 3000)
def rundictiiits():
    runner,globalcache,mdict=get_tester();
    err = 0;
    tot=0;
    for i in range(3000):
        tot+=1;
        with open("/home/lasercat/ssddata/iiit5kdicted/%d.gt" % i, "r") as fp:
            gts = [k.strip() for k in fp];
        try:
            res = dicted_test("/home/lasercat/ssddata/iiit5kdicted/%d.jpg" % i,
                              runner,
                              globalcache, "/run/media/lasercat/ssddata/iiit5kdicted/%d.lexs" % i);
            if (res != gts[0]):
                err += 1
                print(gts, "vs", res, "at", i);
        except:
            err += 1
    print(1 - err / tot)
def rundictsvt():
    runner,globalcache,mdict=get_tester();
    err = 0;
    for i in range(647):

        with open("/run/media/lasercat/ssddata/svtdicted/%d.gt" % i, "r") as fp:
            gts = [k.strip() for k in fp];
        try:
            res = dicted_test("/run/media/lasercat/ssddata/svtdicted/%d.jpg" % i,
                              runner,
                              globalcache, "/run/media/lasercat/ssddata/svtdicted/%d.lex" % i);
            if (res != gts[0]):
                err += 1
                print(gts, "vs", res, "at", i);
        except:
            err += 1
    print(1 - err / 647)

from neko_sdk.ocr_modules.charset.symbols import symbol,latinsym
s=symbol.union(latinsym)
# ic03=1c13-sym-len<3
def ic03_or13(ic03=True):
    runner,globalcache,mdict=get_tester();
    err = 0;
    tot=0;
    onlyi3=0
    for i in range(1107):
        with open("/home/lasercat/ssddata/ic03dicted/%d.gt" % i, "r") as fp:
            gts = [k.strip() for k in fp];
        if(ic03 and (len(gts[0])<=2 or set(gts[0]).intersection(s))):
            onlyi3+=1;
            print(gts[0])
            continue;
        tot+=1;
        try:
            res = dicted_test("/run/media/lasercat/ssddata/ic03dicted/%d.jpg" % i,
                              runner,
                              globalcache, "/run/media/lasercat/ssddata/ic03dicted/%d.lex" % i);
            if (res.lower() != gts[0].lower()):
                err += 1
                print(gts, "vs", res, "at", i);
        except:
            err += 1
    print(tot)
    print(1 - err / tot)

if __name__ == '__main__':
    ic03_or13();
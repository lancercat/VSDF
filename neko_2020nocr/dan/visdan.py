import cv2;
import os;
import shutil;
import numpy as np;
import torch;

def imname(path,id):
    return os.path.join(path, str(id) + "_img.jpg");

def stackname(path,id):
    return os.path.join(path, str(id) + "_img.jpg");

def att_name(path,id,t):
    return os.path.join(path, str(id) + "att_" + str(t) + ".jpg");
def resname(path,id):
    return os.path.join(path, str(id) + "_res.txt")


class visdan:
    def __init__(this,path):
        this.path=path;
        # shutil.rmtree(path,True);
        os.makedirs(path,exist_ok=True);
        this.counter=0;

    def write(this,id,im,A,label,out):
        impath=imname(this.path,id);
        rec_path=resname(this.path,id);
        cv2.imwrite(impath,(im*255)[0].detach().cpu().numpy().astype(np.uint8));
        with open(rec_path,"w+") as fp:
            fp.write(label+"\n");
            fp.write(out + "\n");

        for i in range(min(len(label)+3,A.shape[0])):
            att_mask_name = att_name(this.path,id,i);
            cv2.imwrite(att_mask_name,(A[i]*200*im+56*im)[0].detach().cpu().numpy().astype(np.uint8));
    def writegt(this,id,label,out):
        rec_path=resname(this.path,id);
        with open(rec_path,"w+") as fp:
            fp.write(label+"\n");
            fp.write(out + "\n");


    def add_image(this,ims,label,out,names):
        bs = ims[0].shape[0];
        for bid in range(bs):
            for i in range(len(ims)):
                impath = imname(this.path, str(this.counter)+names[i]);
                try:
                    cv2.imwrite(impath, (ims[i][bid] * 255).permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8));
                except:

                    cv2.imwrite(impath, (ims[i][bid] * 255)[0].detach().cpu().numpy().astype(np.uint8));

            this.writegt(this.counter, label[bid], out[bid]);
            this.counter += 1;


        pass;
    def addbatch(this,im,A,label,out):
        bs=im.shape[0];
        att_masks = torch.nn.functional.interpolate(A, [im.shape[2], im.shape[3]], mode='bilinear');

        for bid in range(bs):
            this.write(this.counter,im[bid],att_masks[bid],label[bid],out[bid]);
            this.counter+=1;
        pass;
#
# def build_stacked_att(path,id,acnt):
#     with open(resname(path,id),"r") as fp:
#         lines=[i.strip() for i in fp ];
#     gt,label=
#
# def compile_dan(path,cnt,acnt,dst):
#     for i in
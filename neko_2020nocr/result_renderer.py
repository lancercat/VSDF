import os;
import numpy as np
import torch;
import shutil;
import cv2
import pylcs;
import regex;
def render_imgschscore(dict,im_file,t,dstfile):
    im = cv2.imread(im_file);
    im = cv2.resize(im, (128, 64))
    with open(t, "r") as gtf:
        gt, ch = [l.strip() for l in gtf];
        if (ch != ""):
            predp = dict["protos"][dict["label_dict"][ch]][0][0].numpy()
            predp = cv2.resize(predp, (32, 32)).astype(np.uint8)
        else:
            predp = np.zeros([32, 32]).astype(np.uint8);

        gtp = dict["protos"][dict["label_dict"][gt]][0][0].numpy()
        gtp = cv2.resize(gtp, (32, 32)).astype(np.uint8);
        im[:32, -32:, :] = gtp.reshape(32, 32, 1);
        im[32:, -32:, :] = 0
        if (gt == ch):
            im[32:, -32:, 1] = predp;
        else:
            im[32:, -32:, 2] = predp;
        cv2.imwrite(dstfile, im);
        pass;

def render_imgschs(dict,im_file,txtfiles):
    for t in txtfiles:
        dim=t.replace("txt","jpg");
        render_imgschscore(dict,im_file,t,dim)

def color(dict,ch,b,g,r):
    if(ch not in dict["label_dict"]):
        default = np.zeros([32, 32, 3], np.uint8) + 255;
        if ch=="⑨" or ch=='[UNK]':
            default[:,:,0]=249;
            default[:, :, 1] = 139;
            default[:, :, 2] = 111;
        else:
            pass;
            # print(ch);
        return default;
    if ch == "[s]" or ch == '[UNK]':
        default = np.zeros([32, 32, 3], np.uint8) + 255;
        default[:, :, 0] = 249;
        default[:, :, 1] = 139;
        default[:, :, 2] = 111;
        return default;
    chid=dict["label_dict"][ch];
    proto=dict["protos"][chid][0][0].numpy();
    ret=np.zeros([proto.shape[0],proto.shape[1],3]);
    ret[:,:,0]=proto*b/255.;
    ret[:,:,1]=proto*g/255.;
    ret[:,:,2]=proto*r/255.;
    return ret.astype(np.uint8);

def render_card(lim,patchs,length):
    res = np.zeros((32, length * 32 + 48, 3));
    res[0:32, 0:48,:] = lim.reshape((32,48,-1));
    for i in range(len(patchs)):
        res[0:32, 48 + i * 32:80 + i * 32] = patchs[i];
    return res;

def render_chars(lim,dict,chs,colors,length):
    patchs=[];
    for i in range(len(chs)):
        patchs.append(color(dict,chs[i],*(colors[i])));
    return render_card(lim,patchs,length);

def dye(ids,srcdst,color):
    for i in ids:
        srcdst[i]=color;
    return srcdst;
def get_unpadded(im,length,kar=False):
    try:
        h=64;
        pad = np.min(np.nonzero(np.hstack(im.sum(0).sum(1))));
        if(pad==0):
            imc=im;
        else:
            imc = im[:, pad:-pad, :]
        if(kar):
            h=int(imc.shape[0]*(length/imc.shape[1]))
        im1 = cv2.resize(imc, (length, h))
    except:
        im1 = cv2.resize(im, (length , 64))
    return im1;

def render_diff(dict,top,bottom,gt,im,sc=(255,255,255),dc=(0,0,255)):


    same_id_top=pylcs.lcs_str2id(bottom,top);
    same_id_bottom=pylcs.lcs_str2id(top,bottom);
    length=max(len(bottom),len(top),len(gt));

    tc=[dc for i in range(len(top))];
    bc = [dc for i in range(len(bottom))];
    gc=[sc for i in range(len(gt))];
    dye(same_id_top,tc,sc);
    dye(same_id_bottom, bc, sc);
    ti=render_chars(np.zeros([32,48],np.uint8),dict,top,tc,length);
    bi = render_chars(np.zeros([32, 48], np.uint8), dict, bottom, bc, length);
    gi = render_chars(np.zeros([32, 48], np.uint8), dict, gt, gc, length);
    im=get_unpadded(im,length*32);
    dst=np.zeros([64+32*3,length*32+48,3]);
    dst[:64,48:]=im;
    dst[64:]=np.concatenate([gi,ti,bi],axis=0);
    return dst;
def render_pred(dict,gt_word,pred_word,PRED):

    if(PRED is None):
        PRED = np.zeros([32, 48, 3], np.uint8);
        if(gt_word==pred_word):
            PRED[:,:,1]=255;
        else:
            PRED[:,:,2]=255;
    pred_patchs=[PRED];
    flag=True;
    if(gt_word is not None):
        corids=pylcs.lcs_str2id(gt_word, pred_word)
    else:
        flag=False;
        corids=set(range(len(pred_word)))

    chs=list(regex.findall(r'\X', pred_word, regex.U))
    if (len(pred_word) != len( chs)):
        flag = False;
        corids = set(range(len(pred_word)))
    for i in range(len(chs)):
        if (i in corids):
            if(flag):
                proto = color(dict, chs[i], 0, 255, 0)
            else:
                proto = color(dict, chs[i], 255, 255, 255)
        else:
            proto = color(dict, chs[i], 0, 0, 255);
        pred_patchs.append(proto);
    return pred_patchs;

def render_gt(dict,seenchs,gt_word,GT):
    gtpatchs=[GT];
    for i in range(len(gt_word)):
        if (gt_word[i] in seenchs):
            proto = color(dict, gt_word[i], 255, 255, 255)
        else:
            proto = color(dict, gt_word[i], 0, 255, 255);
        gtpatchs.append(proto);
    return gtpatchs;

def render_words(dict,seenchs,im,gt_word,pred_words,flag=0):

    PRED = None;
    GT=np.zeros([32,48,3],np.uint8);
    GT[:,:,:]=255;
    try:
        patchs = [
        ]
        if(gt_word is not None):
            patchs.append(render_gt(dict,seenchs,gt_word,GT))
        for word in pred_words:
            patchs.append(render_pred(dict,gt_word,word,PRED));
        # remove flag block
        ml=max([len(i)-1 for i in patchs]);
        taw=ml*32+48

        res=np.zeros((32*len(patchs),taw,3));
        for r in range(len(patchs)):
            off=0;
            for c in range(len(patchs[r])-flag):
                p=patchs[r][c];
                pw=p.shape[1];
                res[r*32:r*32+32,off:off+pw]=p;
                off+=pw;
        if(gt_word is not None):
            lgtp=len(regex.findall(r'\X', gt_word, regex.U))-flag;
        else:
            lgtp=len(regex.findall(r'\X', pred_words[0], regex.U))-flag;
        im1=get_unpadded(im,lgtp*32)
        im11=np.zeros((64,ml*32+48,3));
        im11[:,48:lgtp*32+48,:]=im1;
        fin=np.concatenate([im11.astype(np.uint8),res.astype(np.uint8)],0);
    except:
        print("error during visualization")
        fin=np.zeros([32,32,3]);
    if(gt_word is not None):
        return fin,\
               [1-(pylcs.edit_distance(pred_word,gt_word))/len(gt_word)
                for pred_word in pred_words];
    else:
        return fin,[0 for pred_word in pred_words];

def render_word(dict,seenchs,im,gt_word,pred_word,flag=0):
    fin,neds=render_words(dict,seenchs,im,gt_word,[pred_word],flag);
    return fin,neds[0];
if __name__ == '__main__':
    from neko_sdk.ocr_modules.charset.chs_cset import t1_3755;
    from neko_sdk.ocr_modules.charset.etc_cset import latin62;
    cs=t1_3755.union(latin62);
    for i in range(2000):
        img=cv2.imread("/media/lasercat/ssddata-c3/tmp/"+str(i)+"_img.jpg");
        with open("/media/lasercat/ssddata-c3/tmp/"+str(i)+"_res.txt","r") as ifp:
            [gt,pr]=[i.strip() for i in ifp];
        dict=torch.load("/media/lasercat/backup/deployedlmdbs/dicts/dab62cased.pt")
        red,ned=render_word(dict,cs,img,gt.lower(),pr.lower());
        cv2.imshow("red",red);
        cv2.waitKey(0);
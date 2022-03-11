import glob
import os
import random
import shutil

import cv2
import numpy as np
import torch

from neko_sdk.ocr_modules.charset.chs_cset import t1_3755;
from neko_sdk.ocr_modules.charset.etc_cset import latin62;
from neko_sdk.root import find_export_root,find_model_root
import editdistance as ed
from neko_2020nocr.dan.utils import Loss_counter,neko_os_Attention_AR_counter

from neko_sdk.ocr_modules.charset.jap_cset import hira,kata,Kyoiku_kanji,Joyo_kanji;
def with_hirakata(gt):
    return len(set(gt).intersection(hira.union(kata)))>0;
def wo_hirakata(gt):
    return len(set(gt).intersection(hira.union(kata))) ==0;
def seen(gt):
     return len(set(gt).intersection(t1_3755.union(latin62)))==len(set(gt))
def ukanji(gt):
    return wo_hirakata(gt) and not seen(gt);
def all_words(gt):
    return True;
filters={
    "Overall": all_words,
    "Seen":seen,
    "Unique Kanji": ukanji,
    "All Kanji": wo_hirakata,
    "Kana": with_hirakata
}
import pylcs
def getres(file):
    with open(file,"r") as fp:
        [gt,pr]=[i.strip() for i in fp ][:2];
        return gt, pr;
def accrfolder(root,filter,dst,thresh):
    files=glob.glob(os.path.join(root,"*.txt"));
    tot =0;
    corr=0;
    tned=0;
    arcntr=neko_os_Attention_AR_counter(root,case_sensitive=False);
    tlen=0
    for f in files:
        gt,pr=getres(f);
        tlen+=len(gt);
        if(not filter(gt)):
            continue;
        arcntr.add_iter([pr],[gt],[gt])
        ned=1-pylcs.edit_distance(gt, pr) / len(gt)
        for t in thresh:
            try:
                if(ned*10+9e-9>=t):
                    dfolder=os.path.join(dst,str(t));
                    shutil.copy(f,dfolder);
                    shutil.copy(f.replace(".txt",".jpg"),dfolder);
                    break
                    pass;
            except:
                pass;
        tned+=ned
        if (gt==pr):
            corr+=1;
        tot+=1;
    arcntr.show();
    return corr / tot, tned / tot, corr, tot, tlen / tot;

def maketex(root,name,ks,threshs):
    rd={};
    tex=name
    for k in ks:
        dst=os.path.join(root,k)
        try:
            shutil.rmtree(dst);
        except:
            pass;
        os.makedirs(os.path.join(root,k),exist_ok=True);
        for t in threshs:
            os.makedirs(os.path.join(root, k,str(t)), exist_ok=True);
        if(k not in filters):
            tex+="&";
            continue;
        acr,ccr,corr,tot,alen=accrfolder(root,filters[k],dst,threshs);
        print(k, ccr,acr);
        tex+="&"+"{:.2f}".format(ccr*100)+"/"+"{:.2f}".format(acr*100);
    tex+="\\\\";
    print(tex);


#
def remix(root,pkeys,dwidth=1024,mar=128,flag=44,rows=1,extra_terms=1):
    idict={}
    for k in pkeys:
        images=[];
        images_=glob.glob(os.path.join(root,k,"*.jpg"));
        for i in images_:
            if(i.find("att")!=-1):
                continue;
            else:
                images.append(i);
        random.shuffle(images);
        idict[k]=images;
    a=[]
    for r in range(rows):
        cw = 0;
        imgs = [];
        while cw<dwidth:
            rk=random.sample(pkeys,1)[0];
            if(len(idict[rk])==0):
                continue;
            # a third term may appear beyond visual context
            im=cv2.imread(idict[rk][0])[:128+32*extra_terms,flag:];
            del idict[rk][0];
            if(im.shape[1]+cw<dwidth+mar):
                imgs.append(im);
                cw+=im.shape[1];
        im=cv2.resize(np.concatenate(imgs,1),(dwidth,128));
        a.append(im);
    return np.concatenate(a,axis=0);
def makejp(root,dw=1024):
    thresh=[10,8,5,3,0];
    KS=    "Overall","Speed","Seen","Unique Kanji", "All Kanji","Kana";
    maketex(root,"Ours",KS,thresh);

    G=["10"];
    B=["8","5","3","0"];
    K=["Kana","Unique Kanji","Seen"];
    rs=[]
    for k in K:
        gr=remix(os.path.join(root,k),G,dwidth=dw);
        br=remix(os.path.join(root,k),B,dwidth=dw);
        rs.append(gr);
        rs.append(br);
        # rs.append(br);
    cv2.imwrite(os.path.join(root,"sample.jpg"),np.concatenate(rs))
def makejpg(root,dw=1024,extra_terms=1,rows=2):
    root=os.path.join(root,"JAP_lang")
    thresh=[10,8,5,3,0];
    KS=    "Overall","Speed","Seen","Unique Kanji", "All Kanji","Kana";
    maketex(root,"Ours",KS,thresh);

    G=["10"];
    B=["8","5","3","0"];
    K=["Kana","Unique Kanji","Seen"];
    rs=[]
    for k in K:
        for r in range(rows):
            gr=remix(os.path.join(root,k),G,dwidth=dw,extra_terms=extra_terms);
            # br=remix(os.path.join(root,k),B,dwidth=dw);
            rs.append(gr);
        # gr=remix(os.path.join(root,k),G,dwidth=dw,extra_terms=extra_terms);
        # rs.append(gr);

        # rs.append(br);
    cv2.imwrite(os.path.join(root,"samplegood.jpg"),np.concatenate(rs))
def makejpb(root,dw=1024,extra_terms=1,rows=2):
    root=os.path.join(root,"JAP_lang")

    thresh=[10,8,5,3,0];
    KS=    "Overall","Speed","Seen","Unique Kanji", "All Kanji","Kana";
    maketex(root,"Ours",KS,thresh);

    G=["10"];
    B=["8","5","3","0"];
    K=["Kana","Unique Kanji","Seen"];
    rs=[]
    for k in K:
        for r in range(rows):
        # gr=remix(os.path.join(root,k),G,dwidth=dw);
            br=remix(os.path.join(root,k),B,dwidth=dw,extra_terms=extra_terms);
            rs.append(br);
        # br=remix(os.path.join(root,k),B,dwidth=dw,extra_terms=extra_terms);
        # rs.append(br);

        # rs.append(br);
    cv2.imwrite(os.path.join(root,"samplebad.jpg"),np.concatenate(rs))

def make_close_set(root):
    DS=["CUTE","IC03","IC13","IIIT5k","SVT"];# ,"SVTP"
    thresh=[0];
    A=["0"]
    rs=[];
    for d in DS:
        srcdst=os.path.join(root,d);
        for t in thresh:
            ddir=os.path.join(root, d, str(t))
            shutil.rmtree(ddir,ignore_errors=True);
            os.makedirs(ddir, exist_ok=True);
        accrerfolder(srcdst,filters["Overall"],srcdst,thresh)
        rs.append(remix(os.path.join(root,d),A));
    cv2.imwrite(os.path.join(root,"samplel.jpg"),np.concatenate(rs));
# make_close_set("/run/media/lasercat/ssddata/cvpr22_candidata/8e2/dump/");
def make_krold(root):
    DS=["KR_lang"];# ,"SVTP"
    thresh=[10,8,5,3,0];
    G = ["10"];
    B = ["8", "5", "3", "0"];
    rs=[];
    for d in DS:
        srcdst=os.path.join(root,d);
        for t in thresh:
            ddir=os.path.join(root, d, str(t))
            shutil.rmtree(ddir,ignore_errors=True);
            os.makedirs(ddir, exist_ok=True);
        accrfolder(srcdst,filters["Overall"],srcdst,thresh)
        rs.append(remix(os.path.join(root,d),G,768,rows=4));
    cv2.imwrite(os.path.join(root,"samplegood.jpg"),np.concatenate(rs));
    rs=[];
    for d in DS:
        srcdst=os.path.join(root,d);
        for t in thresh:
            ddir=os.path.join(root, d, str(t))
            shutil.rmtree(ddir,ignore_errors=True);
            os.makedirs(ddir, exist_ok=True);
        accrfolder(srcdst,filters["Overall"],srcdst,thresh)
        rs.append(remix(os.path.join(root,d),B,256,rows=4));
    cv2.imwrite(os.path.join(root,"samplebad.jpg"),np.concatenate(rs));

def make_krg(root,width=1024,extra_terms=1,rows=2):
    DS=["KR_lang"];# ,"SVTP"
    thresh=[10,8,5,3,0];
    G = ["10"];
    B = ["8", "5", "3", "0"];
    rs=[];
    for d in DS:
        srcdst=os.path.join(root,"KR_lang");
        for t in thresh:
            ddir=os.path.join(root, d, str(t))
            shutil.rmtree(ddir,ignore_errors=True);
            os.makedirs(ddir, exist_ok=True);
        accrfolder(srcdst,filters["Overall"],os.path.join(root, d),thresh)
        rs.append(remix(os.path.join(root, d),G,width,rows=rows,extra_terms=extra_terms));
    cv2.imwrite(os.path.join(root,"samplegood.jpg"),np.concatenate(rs));


def make_krb(root,width=1024,extra_terms=1,rows=2):
    DS=["KR_lang"];# ,"SVTP"
    thresh=[10,8,5,3,0];
    G = ["10"];
    B = ["8", "5", "3", "0"];
    rs=[];
    for d in DS:
        srcdst=os.path.join(root,"KR_lang");
        for t in thresh:
            ddir=os.path.join(root, d, str(t))
            shutil.rmtree(ddir,ignore_errors=True);
            os.makedirs(ddir, exist_ok=True);
        accrfolder(srcdst,filters["Overall"],os.path.join(root, d),thresh)
        rs.append(remix(os.path.join(root, d),B,width,rows=rows,extra_terms=extra_terms));
    cv2.imwrite(os.path.join(root,"samplebad.jpg"),np.concatenate(rs));

def test(file):
    with open(file,"r") as fp:
        gt,pred,_=[i.strip() for i in fp];
    return gt==pred;

def get_acr_stat(root,d,S,id):
    "500/closeset_benchmarks/HWDB_unseen/"
    txts=[os.path.join(root,s,"closeset_benchmarks",d,str(id)+".txt") for s in S];
    imgs = [os.path.join(root, s, "closeset_benchmarks", d, str(id) + ".jpg") for s in S];
    stats=[int(test(s)) for s in txts];
    for i in range(1, len(imgs)):
        if(stats[i]<stats[i-1]):
            return None;
    ims=[cv2.imread(img)[:,48:] for img in imgs];
    for i in range(1,len(ims)):
        ims[i]=ims[i][96:]
    return np.concatenate(ims,axis=0)

def make_ch(root):
    DS=["HWDB_unseen","CTWCH_unseen"];
    TN=["base_hwdb_prototyper","base_ctw_prototyper"];
    S=["500","1000","1500","2000"]# ,"SVTP"

    a=list(range(59770));
    random.shuffle(a);
    idx=0;
    t=32;
    dims=[];
    for d in DS:
        rs = [];
        for r in range(4):
            rims=[];
            c=0;
            while c<t:
                I=get_acr_stat(root, d, S, a[idx]);
                idx+=1
                if(I is None):
                    continue;
                c+=1;
                rims.append(I)
            rs.append(np.concatenate(rims,axis=1));
        dims.append(np.concatenate(rs));
        print(idx)
    cv2.imwrite(os.path.join(root,"samplech.jpg"),np.concatenate(dims,axis=1));



def get_acr_stat2(root,d,t,S,id):
    "1000/closeset_benchmarks/base_ctw_prototyper/"
    "500/closeset_benchmarks/HWDB_unseen/"
    txts=[os.path.join(root,"jtrmodels"+s,"closeset_benchmarks",t,d,str(id)+".txt") for s in S];
    imgs = [os.path.join(root, "jtrmodels"+s, "closeset_benchmarks",t, d, str(id) + ".jpg") for s in S];
    stats=[int(test(s)) for s in txts];
    for i in range(1, len(imgs)):
        if(stats[i]<stats[i-1]):
            return None;
    ims=[cv2.imread(img)[:128,48:] for img in imgs];
    for i in range(1,len(ims)):
        ims[i]=ims[i][96:]
    return np.concatenate(ims,axis=0)

def make_ch2(root):
    DS=["HWDB_unseen","CTWCH_unseen"];
    TN=["base_hwdb_prototyper","base_ctw_prototyper"];
    S=["500","1000","1500","2000"]# ,"SVTP"

    a=list(range(59770));
    random.shuffle(a);
    idx=0;
    t=32;
    dims=[];
    for did in range(len(DS)):
        rs = [];
        for r in range(4):
            rims=[];
            c=0;
            while c<t:
                I=get_acr_stat2(root, DS[did],TN[did], S, a[idx]);
                idx+=1
                if(I is None):
                    continue;
                c+=1;
                rims.append(I)
            rs.append(np.concatenate(rims,axis=1));
        dims.append(np.concatenate(rs));
        print(idx)
    cv2.imwrite(os.path.join(root,"samplech.jpg"),np.concatenate(dims,axis=1));

# make_ch2("/run/media/lasercat/ssddata/ijcai22_candidates/g2/DUAL_ch_asc_Odancukmk7hnp_r45_C_trinorm_dsa3_va9r_lsct3sp_2x/");
methods=[
    find_export_root()+"/DUAL_a_Odancukmk7hdtfnp_r45_C_trinorm_dsa3/jtrmodels/closeset_benchmarks/",
    find_export_root()+"/DUAL_a_Odancukmk7hnp_r45_C_trinorm_dsa3/jtrmodels/closeset_benchmarks/",
    find_export_root()+"/DUAL_a_Odancukmk8ahdtfnp_r45_C_trinorm_dsa3/jtrmodels/closeset_benchmarks/",
]
for m in methods:
    makejpg(m,960,extra_terms=0,rows=1);
    makejpb(m,480,extra_terms=0,rows=1);
    make_krg(m,960,extra_terms=0,rows=1);
    make_krb(m,480,extra_terms=0,rows=1);

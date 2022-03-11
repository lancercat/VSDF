# coding:utf-8

from __future__ import print_function

from dicted_eval_configs import dan_mjst_dict_eval_cfg
from neko_2021_mjt.lanuch_std_test import testready
import os,glob,cv2,pylcs,torch
from neko_2020nocr.result_renderer import render_word
from neko_sdk.root import find_export_root,find_model_root,find_data_root

def img_test(img,runner,args):
    res=runner.test_img(0,img,args);
    return  res;
def get_tester(temeta):
    import sys
    if(len(sys.argv)<2):
        argv = ["Meeeeooooowwww",
                find_export_root()+"//DUAL_a_Odancukmk8ahdtfnp_r45pttpt_C_trinorm_dsa3/jtrmodels",
                "_E0",
                find_model_root()+"//DUAL_a_Odancukmk8ahdtfnp_r45pttpt_C_trinorm_dsa3/jtrmodels",
                ]
    else:
        argv=sys.argv;
    return testready(argv,dan_mjst_dict_eval_cfg,temeta)
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

def run_folder(ptfile,sfolder,dfolder,save_root):
    runner,globalcache,mdict=get_tester(ptfile);
    # args = runner.testready();
    # mdict = torch.load(ptfile)

    files=glob.glob(os.path.join(sfolder,"*.jpg"));
    for i in range(len(files)):
        res = img_test(files[i],
                              runner,
                              globalcache);
        base=os.path.basename(files[i]);
        dstt=os.path.join(dfolder,base.replace("jpg","txt"));
        dsti=os.path.join(dfolder,base);
        dim,_=render_word(mdict,{},cv2.imread(files[i]),None,res,0);
        cv2.imwrite(dsti,dim);
        print(res);
        with open(dstt,"w+") as fp:
            fp.writelines(res);




if __name__ == '__main__':
    run_folder(find_data_root()+"/dicts/dabgreek.pt",
               find_data_root()+"/greek/randomdata/",
               find_data_root()+"/greek/randomdata/open_res/",
               find_model_root()+"//DUAL_a_Odancukmk8ahdtfnp_r45pttpt_C_trinorm_dsa3/jtrmodels");

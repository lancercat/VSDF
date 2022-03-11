import os
from neko_2020nocr.dan.dataloaders.dataset_scene import colored_lmdbDataset_repeatHS,colored_lmdbDataset
from neko_2020nocr.dan.methods_pami.pami_osds_paths  import get_lsvtK_path,get_ctwK_path,\
    get_mlt_chlatK_path,get_artK_path,get_rctwK_path,get_mltjp_path,get_monkey_path,get_mltkr_path
from torchvision import transforms
from neko_2021_mjt.dataloaders.sampler import randomsampler
def get_chs_tr_meta(root):
    trmeta = os.path.join(root,"dicts","dab3791WT.pt");
    return  trmeta;
def get_chs_sc_meta(root):
    trmeta = os.path.join(root,"dicts","dab3791SC.pt");
    return  trmeta;
def get_jap_te_meta(root):
    temeta = os.path.join(root,"dicts","dabjpmlt.pt");
    return  temeta;
def get_chs_tr_metag2(root):
    trmeta = os.path.join(root,"dicts","dab3791WTg2.pt");
    return  trmeta;
def get_jap_te_metag2(root):
    temeta = os.path.join(root,"dicts","dabjpmltg2.pt");
    return  temeta;
def get_chs_tr_meta64(root):
    trmeta = os.path.join(root,"dicts","dab3791WT64.pt");
    return  trmeta;
def get_jap_te_meta64(root):
    temeta = os.path.join(root,"dicts","dabjpmlt64.pt");
    return  temeta;
def get_jap_te_metaosr(root):
    temeta = os.path.join(root,"dicts","dabjpmltch_seen.pt");
    return  temeta;
def get_jap_te_metagosr(root):
    temeta = os.path.join(root,"dicts","dabjpmltch_nohirakata.pt");
    return  temeta;

def get_kr_te_meta(root):
    temeta = os.path.join(root,"dicts","dabkrmlt.pt");
    return  temeta;
def get_eval_kr_color(root,maxT,hw=[32,128]):
    teroot= get_mltkr_path(root);
    return {
        "dict_dir":None,
        "case_sensitive": False,
        "te_case_sensitive": False,
            "datasets":{
                "KR_lang": get_eval_word_color_core(teroot,maxT,hw),
            }
        }
def get_chs_HScqa(root,maxT,bsize=48,rep=1,hw=[32,128]):
    if(rep>=0):
        dla={
                'batch_size': bsize,
                'shuffle': True,
                'num_workers': 8,
            }
    elif(rep==-1):
        dla = {
            'batch_size': bsize,
            "sampler": randomsampler(None),
            'num_workers': 8,
        }

    return \
    {

        "type": colored_lmdbDataset_repeatHS,
        'ds_args':
        {
            "repeat": rep,
            'roots': [get_lsvtK_path(root),
                      get_ctwK_path(root),
                      get_mlt_chlatK_path(root),
                      get_artK_path(root),
                      get_rctwK_path(root)
                      ],
            'img_height': hw[0],
            'img_width': hw[1],
            'transform': transforms.Compose([transforms.ToTensor()]),
            'global_state': 'Train',
            "maxT": maxT,
            "qhb_aug": True
        },
        'dl_args':dla,
    }

def get_eval_word_color_core(teroot,maxT,hw=[32,128]):
    return \
    {
        "type": colored_lmdbDataset,
        'ds_args':
        {
            'roots': [teroot],
            'img_height': hw[0],
            'img_width': hw[1],
            'transform': transforms.Compose([transforms.ToTensor()]),
            'global_state': 'Test',
            "maxT": maxT
        },
        'dl_args':
        {
            'batch_size': 16,
            'shuffle': False,
            'num_workers': 5,
        },
    }
def get_eval_monkey_color(root,maxT,lang,hw=[32,128]):
    teroot= get_monkey_path(root,lang);
    return {
        "dict_dir":None,
        "case_sensitive": False,
        "te_case_sensitive": False,
            "datasets":{
                "MONKEY_"+lang: get_eval_word_color_core(teroot,maxT,hw),
            }
        }
def get_eval_jap_color(root,maxT,hw=[32,128]):
    teroot= get_mltjp_path(root);
    return {
        "dict_dir":None,
        "case_sensitive": False,
        "te_case_sensitive": False,
            "datasets":{
                "JAP_lang": get_eval_word_color_core(teroot,maxT,hw),
            }
        }
def get_train_chs_color(root,maxT,hw=[32,128]):
    artroot= get_artK_path(root);
    return {
        "dict_dir":None,
        "case_sensitive": False,
        "te_case_sensitive": False,
            "datasets":{
                "ART": get_eval_word_color_core(artroot,maxT,hw),
            }
        }

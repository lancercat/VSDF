from neko_2020nocr.dan.dataloaders.dataset_scene import lmdbDataset_repeatS,lmdbDataset
from neko_2020nocr.dan.dataloaders.dataset_scene import colored_lmdbDataset,colored_lmdbDataset_repeatS

from neko_2020nocr.dan.methods_pami.pami_osds_paths import \
    get_stdhwdbtr,get_stdhwdbte,\
    get_stdctwchtr,get_stdctwchte
from torchvision import transforms

def get_hwdb_tr_meta(root,trcnt):
    _, trmeta = get_stdhwdbtr(trcnt, root);
    return  trmeta;
def get_hwdb_te_meta(root):
    _, temeta = get_stdhwdbte(root);
    return  temeta;

def get_chs_hwdbS(root,trcnt,rgb=False,bsize=160):
    rep=1;
    if(trcnt<=700):
        rep=2;
    trroot, _ = get_stdhwdbtr(trcnt,root);
    if (rgb is False):
        dstype = lmdbDataset_repeatS
    else:
        dstype = colored_lmdbDataset_repeatS

    return \
    {
        "type": dstype,
        'ds_args':
        {
            "repeat": rep,
            'roots': [trroot],
            'img_height': 32,
            'img_width': 64,
            'transform': transforms.Compose([transforms.ToTensor()]),
            'global_state': 'Train',
            "maxT": 2
        },
        'dl_args':
        {
            'batch_size': bsize,
            'shuffle': True,
            'num_workers': 15,
        },
    }

def get_eval_chs_hwdbS_core(teroot,rgb=False):
    if(rgb is False):
        dstype=lmdbDataset
    else:
        dstype=colored_lmdbDataset
    return \
    {
        "type": dstype,
        'ds_args':
        {
            'roots': [teroot],
            'img_height': 32,
            'img_width': 64,
            'transform': transforms.Compose([transforms.ToTensor()]),
            'global_state': 'Test',
            "maxT": 2
        },
        'dl_args':
        {
            'batch_size': 8,
            'shuffle': False,
            'num_workers': 5,
        },
    }

def get_eval_chs_hwdbS(root,rgb=False):
    teroot, _ = get_stdhwdbte(root);
    return {

        "dict_dir":None,
        "case_sensitive": False,
        "te_case_sensitive": False,
            "datasets":{
                "HWDB_unseen": get_eval_chs_hwdbS_core(teroot,rgb),
            }
        }

def get_eval_chstr_hwdbS(root,trcnt):
    trroot, _ = get_stdhwdbtr(trcnt, root);
    return {
        "dict_dir": None,
        "case_sensitive": False,
        "te_case_sensitive": False,
        "datasets": {
            "CTWCH_train": get_eval_chs_hwdbS_core(trroot),
        }
    }



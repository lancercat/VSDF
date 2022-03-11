
from neko_2021_mjt.dataloaders.neko_fsl_loader import neko_fsl_no_proto_task_dataset;
from neko_2021_mjt.dataloaders.neko_fsl_loader2 import neko_fsl_no_proto_task_datasetg2
from neko_2021_mjt.dataloaders.neko_fsl_eval_loader import neko_gfsl_eval_task_dataset;
from neko_2021_mjt.dataloaders.sampler import randomsampler
from neko_sdk.ds_meta.oracle_1 import all
import os
from torchvision import transforms

def get_orahw_tr_meta(root):
    return os.path.join(root,"dicts","dab_oracle_tr.pt");
def get_orahw_te_meta(root):
    return os.path.join(root,"dicts","dab_oracle_te.pt");

def get_orahw_train(root):
    trroot = os.path.join(root,"oraclehw");
    rep=1
    return \
    {
        "type": neko_fsl_no_proto_task_datasetg2,
        'ds_args':
        {
            "db_root":trroot,
            "waycnt":64,
            "ucnt": 16,
            "labelset": all[:3400],
            "dsize": [32,32],
        },
        'dl_args':
        {
            'batch_size': 1,
            "sampler":randomsampler(None),
            'num_workers': 5,
        },
    }

def get_eval_orahw_core(root):
    teroot = os.path.join(root,"oraclehw");
    return \
    {
        "type": neko_gfsl_eval_task_dataset,
        'ds_args':
        {
            'db_root': teroot,
            'labelset':  all[3400:],
            'dsize': [32, 32],
        },
        'dl_args':
        {
            'batch_size': 64,
            'shuffle': False,
            'num_workers': 5,
        },
    }
def get_eval_orahw(root):
    return {
        "dict_dir":None,
        "case_sensitive": False,
        "te_case_sensitive": False,
            "datasets":{
                "jrc_traffic_eval": get_eval_orahw_core(root),
            }
        }

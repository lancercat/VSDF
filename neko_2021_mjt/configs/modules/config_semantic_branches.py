from neko_2021_mjt.modulars.dan.neko_sementic_prototyper import neko_sampled_sementic_branch
from neko_2021_mjt.modulars.default_config import get_default_model

def get_sampled_semantic_branch(arg_dict,path,optim_path=None):
    adict = {
        "spks": arg_dict["spks"],
        "feat_ch": arg_dict["feat_ch"],
        "capacity": arg_dict["capacity"],
    }
    return get_default_model(neko_sampled_sementic_branch,adict,path,arg_dict["with_optim"],optim_path)

def config_sampled_semantic_branch(capacity,feat_ch):
    return \
    {
        "save_each": 20000,
        "modular": get_sampled_semantic_branch,
        "args":
            {
                "with_optim": True,
                "spks": ["[s]","[UNK]"],
                "capacity": capacity,
                "feat_ch": feat_ch,
            }
    }

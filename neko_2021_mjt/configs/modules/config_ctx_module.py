import torch
from neko_2021_mjt.modulars.dan.ctxmodules.v2s import neko_basic_ctx_module
from neko_2021_mjt.modulars.default_config import get_default_model

def get_ctxmodule(arg_dict,path,optim_path=None):
    adict={
        "feat_ch": arg_dict["feat_ch"],
        "nhead": arg_dict["nhead"],
        "nlay":arg_dict["nlay"]
    }
    return get_default_model(neko_basic_ctx_module,adict,path,arg_dict["with_optim"],optim_path);

def config_ctxmodule(feat_ch,nhead=8,nlay=4):
    return \
    {
        "save_each": 20000,
        "modular": get_ctxmodule,
        "args":
            {
                "nhead":nhead,
                "nlay": nlay,
                "feat_ch":feat_ch,
                "with_optim": True
            },
    }
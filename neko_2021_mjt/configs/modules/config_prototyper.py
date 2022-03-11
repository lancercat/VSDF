from neko_sdk.ocr_modules.neko_prototyper_gen2.neko_label_sampler import neko_prototype_sampler_basic,neko_prototyper,neko_prototyperR34
from neko_2021_mjt.modulars.default_config import get_default_model

def get_prototyper(arg_dict,path,optim_path=None):
    adict = {
        "spks": arg_dict["spks"],
        "output_channel": arg_dict["output_channel"],
        "capacity": arg_dict["capacity"],
    }
    return get_default_model(neko_prototyper,adict,path,arg_dict["with_optim"],optim_path)

def config_prototyper(capacity,feat_ch):
    return \
    {
        "save_each": 20000,
        "modular": get_prototyper,
        "args":
            {
                "with_optim": True,
                "spks": ["[s]"],
                "capacity": capacity,
                "output_channel": feat_ch,
            }
    }
def get_prototyper_r34(arg_dict,path,optim_path=None):
    adict = {
        "spks": arg_dict["spks"],
        "output_channel": arg_dict["output_channel"],
        "capacity": arg_dict["capacity"],
    }
    return get_default_model(neko_prototyperR34,adict,path,arg_dict["with_optim"],optim_path)

def config_prototyper_r34(capacity,feat_ch):
    return \
    {
        "save_each": 20000,
        "modular": get_prototyper_r34,
        "args":
            {
                "with_optim": True,
                "spks": ["[s]"],
                "capacity": capacity,
                "output_channel": feat_ch,
            }
    }
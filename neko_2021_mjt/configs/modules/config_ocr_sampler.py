import torch
from neko_sdk.ocr_modules.neko_prototyper_gen2.neko_label_sampler import neko_prototype_sampler_basic,neko_prototyper
from neko_2021_mjt.modulars.default_config import get_default_model

def get_meta_holder(arg_dict,path,optim_path=None):
    adict={
        "meta_args": arg_dict["meta_args"],
        "sampler_args": arg_dict["sampler_args"],
    }
    return get_default_model(neko_prototype_sampler_basic,adict,path,arg_dict["with_optim"],optim_path);

def config_ocr_sampler(meta_path,capacity):
    return \
    {
        "save_each": 0,
        "modular": get_meta_holder,
        "args":
            {
                "meta_args":
                    {
                        "meta_path": meta_path,
                        "case_sensitive": False,
                    },
                "sampler_args":
                    {
                        "max_batch_size": capacity,
                        "val_frac": 0.8,
                        "neg_servant": True,
                    },
                "with_optim": False
            },
    }
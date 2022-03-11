from neko_2021_mjt.modulars.dan.decoupled_prototyper_system import neko_prototyper_sp;
from neko_2021_mjt.modulars.default_config import get_default_model
def get_sp_prototyper(arg_dict,path,optim_path=None):
    adict = {
        "spks": arg_dict["spks"],
        "output_channel": arg_dict["output_channel"],
    }
    return get_default_model(neko_prototyper_sp,adict,path,arg_dict["with_optim"],optim_path)

def config_sp_prototyper(feat_ch=512,use_sp=True):
    return \
        {
            "save_each": 20000,
            "modular": get_sp_prototyper,
            "args":
                {
                    "with_optim": use_sp,
                    "spks": ["[s]"],
                    "output_channel": feat_ch,
                },
        }
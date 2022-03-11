from neko_2021_mjt.modulars.default_config import get_default_model
from neko_2021_mjt.modulars.neko_label_stats import neko_pystat;
def get_pystat(arg_dict,path,optim_path=None):
    args={

    };
    # args["num_channels"]=arg_dict["num_channels"]
    return get_default_model(neko_pystat,args,path,arg_dict["with_optim"],optim_path);

def config_vpystat():
    return \
    {
        "modular": get_pystat,
        "save_each": 20000,
        "args":
            {
                "with_optim": False
            },
    }
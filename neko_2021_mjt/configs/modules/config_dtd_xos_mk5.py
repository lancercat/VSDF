from neko_2021_mjt.modulars.dan.neko_CFDTD_mk5 import neko_os_CFDTD_mk5
from neko_2021_mjt.modulars.default_config import get_default_model

def get_dtdmk5_xos(arg_dict,path,optim_path=None):
    args={
    };
    return get_default_model(neko_os_CFDTD_mk5,args,path,arg_dict["with_optim"],optim_path);

def config_dtdmk5():
    return \
    {
        "save_each": 20000,
        "modular": get_dtdmk5_xos,
        "args":
            {
                "with_optim": False
            },
    }

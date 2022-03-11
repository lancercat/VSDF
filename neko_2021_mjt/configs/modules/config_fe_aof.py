from neko_2020nocr.dan.configs.pipelines_pami import get_cco_fe_args;
from neko_2021_mjt.modulars.default_config import get_default_model

from neko_2021_mjt.modulars.dan.neko_xtra_fe_aof import neko_cco_Feature_Extractor_thicc_std,neko_cco_Feature_Extractor_thicc_stdlr,neko_cco_Feature_Extractor_thicc_sd,neko_cco_Feature_Extractor_thicc_sdlr
def get_cco_thicc_std(arg_dict,path,optim_path=None):
    args=get_cco_fe_args(arg_dict["hardness"],arg_dict["ouch"],arg_dict["ich"]);
    return get_default_model(neko_cco_Feature_Extractor_thicc_std,args,path,arg_dict["with_optim"],optim_path);


def config_fe_cco_thicc_std(ich,feat_ch,input_shape=None):
    return \
    {
        "modular": get_cco_thicc_std,
        "save_each": 20000,
        "args":
            {
                "hardness": 0.5,
                "ich": ich,
                "ouch": feat_ch,
                "with_optim": True,
                "input_shape": input_shape,
                "strides":None
            }
    }

def get_cco_thicc_stdlr(arg_dict,path,optim_path=None):
    args=get_cco_fe_args(arg_dict["hardness"],arg_dict["ouch"],arg_dict["ich"]);
    return get_default_model(neko_cco_Feature_Extractor_thicc_stdlr,args,path,arg_dict["with_optim"],optim_path);


def config_fe_cco_thicc_stdlr(ich,feat_ch,input_shape=None):
    return \
    {
        "modular": get_cco_thicc_stdlr,
        "save_each": 20000,
        "args":
            {
                "hardness": 0.5,
                "ich": ich,
                "ouch": feat_ch,
                "with_optim": True,
                "input_shape": input_shape,
                "strides":None
            }
    }
from neko_2021_mjt.modulars.dan.neko_xtra_fe_aof import neko_cco_Feature_Extractor_thicc_sdlr

def get_cco_thicc_sdlr(arg_dict,path,optim_path=None):
    args=get_cco_fe_args(arg_dict["hardness"],arg_dict["ouch"],arg_dict["ich"]);
    return get_default_model(neko_cco_Feature_Extractor_thicc_sdlr,args,path,arg_dict["with_optim"],optim_path);


def config_fe_cco_thicc_sdlr(ich,feat_ch,input_shape=None):
    return \
    {
        "modular": get_cco_thicc_sdlr,
        "save_each": 20000,
        "args":
            {
                "hardness": 0.5,
                "ich": ich,
                "ouch": feat_ch,
                "with_optim": True,
                "input_shape": input_shape,
                "strides":None
            }
    }
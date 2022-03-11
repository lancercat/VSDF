from neko_2020nocr.dan.configs.pipelines_pami import get_cco_fe_args;
from neko_2021_mjt.modulars.default_config import get_default_model
from neko_2020nocr.dan.dan_modules_pami.neko_xtra_fe import neko_cco_Feature_Extractor
from neko_2020nocr.dan.dan_modules_pami.neko_xtra_fe_xl import neko_Feature_Extractor_thicc, \
    neko_cco_Feature_Extractor_thicc


def get_cco(arg_dict,path,optim_path=None):
    args=get_cco_fe_args(arg_dict["hardness"],arg_dict["ouch"],arg_dict["ich"],expf=arg_dict["expf"]);
    return get_default_model(neko_cco_Feature_Extractor,args,path,arg_dict["with_optim"],optim_path);


def config_fe_cco(ich,feat_ch,input_shape=None,expf=1):
    return \
    {
        "modular": get_cco,
        "save_each": 20000,
        "args":
            {
                "hardness": 0.5,
                "ich": ich,
                "ouch": feat_ch,
                "with_optim": True,
                "input_shape": input_shape,
                "strides":None,
                "expf":expf
            }
    }
def get_cco_thicc(arg_dict,path,optim_path=None):
    args=get_cco_fe_args(arg_dict["hardness"],arg_dict["ouch"],arg_dict["ich"]);
    return get_default_model(neko_cco_Feature_Extractor_thicc,args,path,arg_dict["with_optim"],optim_path);


def config_fe_cco_thicc(ich,feat_ch,input_shape=None):
    return \
    {
        "modular": get_cco_thicc,
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

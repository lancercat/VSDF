from neko_2020nocr.dan.configs.pipelines_pami import get_cco_fe_args;
from neko_2020nocr.dan.DAN import Feature_Extractor
from neko_2021_mjt.modulars.default_config import get_default_model
from neko_2020nocr.dan.configs.pipelines_pami import get_bl_fe_args;

def get_dan_r45(arg_dict,path,optim_path=None):
    args=get_bl_fe_args(arg_dict["ouch"],arg_dict["ich"]);
    return get_default_model(Feature_Extractor,args,path,arg_dict["with_optim"],optim_path);


def config_fe_r45(ich,feat_ch,input_shape=None):
    return \
    {
        "modular": get_dan_r45,
        "save_each": 20000,
        "args":
            {
                "ich": ich,
                "ouch": feat_ch,
                "with_optim": True,
                "input_shape": input_shape,
                "strides":None
            }
    }


def get_std_fe_args(ouch,ich=1,strides=None,input_shape=None):
    if (strides is None):
        strides= [2,2,2,2,2]
    if(input_shape is None):
        input_shape=[ich,64,480];
    else:
        input_shape=[ich,input_shape[0],input_shape[1]];
    return {
            'strides':strides,
            'input_shape': input_shape,
            "ouch":ouch,# C x H x W
        }
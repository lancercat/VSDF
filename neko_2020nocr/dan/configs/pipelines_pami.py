def get_bl_fe_args(nchannel=512,ich=1):
    return  {
            'strides': [(1,1), (2,2), (1,1), (2,2), (1,1), (1,1)],
            'compress_layer' : False,
            'input_shape': [ich, 32, 128],
            "oupch": nchannel,  # C x H x W
    };
def get_cco_fe_args(hardness,nchannel=512,ich=1,strides=None,input_shape=None,expf=1):
    if (strides is None):
        strides= [(1,1), (2,2), (1,1), (2,2), (1,1), (1,1)]
    if(input_shape is None):
        input_shape=[ich,32,128];
    else:
        input_shape=[ich,input_shape[0],input_shape[1]];
    return {
            "expf":expf,
            'strides':strides,
            'compress_layer' : False,
            'input_shape': input_shape, # C x H x W
            "hardness": hardness,
            "oupch": nchannel,
        }
def get_cam_args(maxT,nch=64):
    return  {
            'maxT': maxT,
            'depth': 8,
            'num_channels': nch,
        }
def get_bl_dtd_args(nclass):
    return {
            'nclass': nclass, # extra 2 classes for Unkonwn and End-token
            'nchannel': 512,
            'dropout': 0.3,
        };

def get_xos_dtd_args():
    return {
            'nchannel': 512,
            'dropout': 0.3,
        };

def get_pe_args(ptfile):
    return {
        "meta_path": ptfile,
        "nchannel": 512,
        'case_sensitive': False,
    }

def armtoken_bl(d,token,prefix):
    if (token is None):
        #
        d['init_state_dict_fe'] = None;
        d['init_state_dict_cam'] = None;
        d['init_state_dict_dtd'] = None;
    else:
        d['init_state_dict_fe'] = "../../models/scene/" + prefix + "_" + token + "_M0.pth";
        d['init_state_dict_cam'] = "../../models/scene/" + prefix + "_" + token + "_M1.pth";
        d['init_state_dict_dtd'] = "../../models/scene/" + prefix + "_" + token + "_M2.pth";
    return d

def armtoken_xos(d,token,prefix):
    if (token is None):
        #
        d['init_state_dict_fe'] = None;
        d['init_state_dict_cam'] = None;
        d['init_state_dict_dtd'] = None;
        d['init_state_dict_pe']=None;
    else:
        d['init_state_dict_fe'] = "../../models/scene/" + prefix + "_" + token + "_M0.pth";
        d['init_state_dict_cam'] = "../../models/scene/" + prefix + "_" + token + "_M1.pth";
        d['init_state_dict_dtd'] = "../../models/scene/" + prefix + "_" + token + "_M2.pth";
        d['init_state_dict_pe'] = "../../models/scene/" + prefix + "_" + token + "_M3.pth";
    return d


def _get_baseline(FE,CAM,DTD,nclass,maxT):
    return   {
        'FE': FE,
        'FE_args': get_bl_fe_args(),
        'CAM': CAM,
        'CAM_args': get_cam_args(maxT),
        'DTD': DTD,
        'DTD_args': get_bl_dtd_args(nclass)
    }

def _get_baseline_cco(FE,CAM,DTD,nclass,hardness,maxT):
    return   {
        'FE': FE,
        'FE_args': get_cco_fe_args(hardness),
        'CAM': CAM,
        'CAM_args': get_cam_args(maxT),
        'DTD': DTD,
        'DTD_args':get_bl_dtd_args(nclass)
    }

def _get_dos(FE,CAM,DTD,PE,ptfile,maxT):
    return {
        'FE': FE,
        'FE_args': get_bl_fe_args(),
        'CAM': CAM,
        'CAM_args': get_cam_args(maxT),
        'DTD': DTD,
        'DTD_args': get_xos_dtd_args(),
        'PE':PE,
        "PE_args":get_pe_args(ptfile)
    }

def _get_dos_cco(FE,CAM,DTD,PE,ptfile,hardness,maxT):
    return {
        'FE': FE,
        'FE_args': get_cco_fe_args(hardness),
        'CAM': CAM,
        'CAM_args': get_cam_args(maxT),
        'DTD': DTD,
        'DTD_args': get_xos_dtd_args(),
        'PE':PE,
        "PE_args":get_pe_args(ptfile)
    }



def get_baseline(FE,CAM,DTD,CC,prefix,token,maxT=25):
    d=_get_baseline(FE,CAM,DTD,CC,maxT)
    return armtoken_bl(d,token,prefix);

def get_baseline_cco(FE,CAM,DTD,CC,prefix,token,hardness,maxT=25):
    d=_get_baseline_cco(FE,CAM,DTD,CC,hardness,maxT)
    return armtoken_bl(d,token,prefix);

def get_dos(FE,CAM,DTD,PE,ptfile,prefix,token,maxT=25):
    d=_get_dos(FE,CAM,DTD,PE,ptfile,maxT);
    return armtoken_xos(d, token, prefix);

def get_dos_cco(FE,CAM,DTD,PE,ptfile,prefix,token,hardness,maxT=25):
    d=_get_dos_cco(FE,CAM,DTD,PE,ptfile,hardness,maxT);
    return armtoken_xos(d, token, prefix);


from neko_2021_mjt.modulars.dan.danloss import osdanloss,osdanloss_trident,osdanloss_clsemb,fsldanloss_clsembohem,osdanloss_clsctx;
def get_loss(arg_dict,path,optim_path=None):
    mod=osdanloss(arg_dict);
    return mod,None,None;
def config_cls_emb_loss():
    return \
    {
        "save_each": 0,
        "modular": get_loss,
        "args":
            {
                "wcls": 1,
                "wemb": 0.3,
                "wsim": 0,
                "wmar": 0
            },
    }

def get_loss_trident(arg_dict,path,optim_path=None):
    mod=osdanloss_trident(arg_dict);
    return mod,None,None;
def config_cls_emb_loss_trident():
    return \
    {
        "save_each": 0,
        "modular": get_loss_trident,
        "args":
            {
                "wcls": 1,
                "wemb": 0.3,
                "wrew": 1,
                "ppr": 0.3
            },
    }
def get_loss_cls_emb2(arg_dict,path,optim_path=None):
    mod=osdanloss_clsemb(arg_dict);
    return mod,None,None;
def config_cls_emb_loss2(wemb=0.3,reduction=True):
    return \
    {
        "save_each": 0,
        "modular": get_loss_cls_emb2,
        "args":
            {
                "wcls": 1,
                "reduction": reduction,
                "wemb": wemb,
            },
    }

def get_loss_cls_emb_fslohem(arg_dict,path,optim_path=None):
    mod=fsldanloss_clsembohem(arg_dict);
    return mod,None,None;
def config_cls_emb_lossosd(dirty_frac=0.1):
    return \
    {
        "save_each": 0,
        "modular": get_loss_cls_emb_fslohem,
        "args":
            {
                "too_simple_frac": 0,
                "dirty_frac": dirty_frac,
                "wcls": 1,
                "wemb": 0.3,
            },
    }

def config_cls_emb_lossohem(dirty_frac=0.1,too_simple_frac=0.4):
    return \
    {
        "save_each": 0,
        "modular": get_loss_cls_emb_fslohem,
        "args":
            {
                "too_simple_frac": too_simple_frac,
                "dirty_frac": dirty_frac,
                "wcls": 1,
                "wemb": 0,
            },
    }




def config_cls_lossohem(dirty_frac=0.1,too_simple_frac=0.01,wcls=1):
    return \
    {
        "save_each": 0,
        "modular": get_loss_cls_emb_fslohem,
        "args":
            {
                "too_simple_frac": too_simple_frac,
                "dirty_frac": dirty_frac,
                "wcls": wcls,
                "wemb": 0,
            },
    }
def get_loss_clsctx(arg_dict,path,optim_path=None):
    mod=osdanloss_clsctx(arg_dict);
    return mod,None,None;
def config_cls_ctx_loss():
    return \
    {
        "save_each": 0,
        "modular": get_loss_cls_emb2,
        "args":
            {

                "wcls": 0.5,
                "wctx": 0.2,
                "wshr": 0.3,
                "wpemb": 0.3,
                "wgemb": 0.3,
            },
    }

import torch
from torch import optim
def get_default_optim(params,lr=1.0):
    optimizer = optim.Adadelta(params, lr=lr)
    optimizer_sched = optim.lr_scheduler.MultiStepLR(optimizer, [3, 5], 0.1)
    return optimizer, optimizer_sched;

def get_default_model(mod,args,path=None,with_optim=True,optim_path=None,optpara=None):
    if(optpara is None):
        optpara={}
    model = mod(**args);
    optimizer, optimizer_sched = None, None;
    if (with_optim):
        optimizer, optimizer_sched = get_default_optim(model.parameters(),**optpara);
    return model, optimizer, optimizer_sched;

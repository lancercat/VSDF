import torch;
from neko_2021_mjt.debug_and_visualize.laserbeam import neko_Backprop
# we need something to wrap the routine.

class bogomod(torch.nn.Module):
    def __init__(this,visible,moddict):
        super(bogomod,this).__init__();
        this.core=visible;
        mcvt=this.core.mod_cvt_dict;
        tmoddict={}
        for name in mcvt:
            if(type(mcvt[name])==list or mcvt[name]=="NEPnoneNEP"):
                continue;
            tm=moddict[mcvt[name]].get_torch_module_dict();
            if(tm is None):
                continue;
            tmoddict[name] = tm;
        for name in tmoddict:
            this.add_module(name,tmoddict[name]);
    def load(this,input_dict,modular_dict,at_time,bid=0):
        this.t=at_time;
        this.modular_dict=modular_dict;
        this.input_dict=input_dict;
        image=input_dict["image"][bid:bid+1];
        text=this.input_dict["label"][bid][at_time];
        if(text in input_dict["tdict"]):
            raberu=input_dict["tdict"][text];
        else:
            raberu=input_dict["tdict"]["[UNK]"]# Umm, just to distinguish it from ascii encoding
        return image,text,raberu;
    def forward(this,image):
        logit=this.core.vis_logit(image,this.input_dict,this.modular_dict,this.t);
        if(logit is None):
            return None;
        return logit.unsqueeze(0);

# cats LOVE laser dots.
class neko_laser:
    def __init__(this,model,moddict):
        this.model=bogomod(model,moddict);
        this.bper=neko_Backprop(this.model,this.model.feature_extractor.shared_fe_0_conv,0)
    def vis_chars(this,input_dict,modular_dict):
        bs=input_dict["image"].shape[0];
        grads=[];
        for bid in range(bs):
            text=input_dict["label"][bid];
            cgs=[];
            for t in range(0,len(text)):
                image,text,raberu=this.model.load(input_dict,modular_dict,t,bid);
                grad_t=this.bper.calculate_gradients(image,raberu,take_max=True,guided=False,use_gpu=True);
                if(grad_t is None):
                    continue;
                cgs.append(grad_t);
            grads.append(cgs)
        return grads;

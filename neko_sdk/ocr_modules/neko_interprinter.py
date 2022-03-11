from  torch import  nn;
# from  torchvision.models import resnet18,resnet34;
from  neko_sdk.encoders.tv_res_nip import resnet18,resnet34;

from neko_sdk.encoders.ocr_networks.neko_pyt_resnet_np import resnet18np,resnet34np;
# from neko_sdk.encoders.feat_networks.ires import conv_iResNet
import torch;

class neko_visual_only_interprinter(nn.Module):
    def __init__(this,feature_cnt,core=None):
        super(neko_visual_only_interprinter,this).__init__();
        if core is None:
            this.core=resnet18(num_classes=feature_cnt);
        else:
            this.core=core;
    def forward(this,view_dict) :
        # vis_proto=view_dict["visual"];
        vp=this.core(view_dict);

        # print(nvp.norm(dim=1))
        return vp

class magic_core(nn.Module):
    def __init__(this,feature_cnt):
        super(magic_core, this).__init__();
        this.c=conv_iResNet([3, 32, 32], [2, 2, 2, 2], [1, 2, 2, 2], [32, 32, 32, 32],
                     init_ds=2, density_estimation=False, actnorm=True);
        this.f=torch.nn.Linear(768,feature_cnt,False)
        this.d=torch.nn.Dropout(0.1);
    def forward(this,x):
        c=this.c(x);
        c=c.mean(dim=(2,3));
        p=this.f(c);
        return this.d(p)

# class neko_visual_only_interprinter_inv(nn.Module):
#     def __init__(this,feature_cnt,core=None):
#         super(neko_visual_only_interprinter_inv,this).__init__();
#         if core is None:
#             this.core=magic_core(feature_cnt)
#         else:
#             this.core=core;
#     def forward(this,view_dict) :
#         # vis_proto=view_dict["visual"];
#         vp=this.core(view_dict);
#
#         # print(nvp.norm(dim=1))
#         return vp
class neko_visual_only_interprinterHD(nn.Module):
    def __init__(this,feature_cnt,core=None):
        super(neko_visual_only_interprinterHD,this).__init__();
        if core is None:
            this.core=resnet18np(outch=feature_cnt);
        else:
            this.core=core;
    def forward(this,view_dict) :
        # vis_proto=view_dict["visual"];
        vp=this.core(view_dict).permute(0,2,3,1).reshape(view_dict.shape[0],-1);
        # print(nvp.norm(dim=1))
        return vp


class neko_visual_only_interprinterR34(nn.Module):
    def __init__(this, feature_cnt, core=None):
        super(neko_visual_only_interprinterR34, this).__init__();
        if core is None:
            this.core = resnet34(num_classes=feature_cnt);
        else:
            this.core = core;


    def forward(this, view_dict):
        # vis_proto=view_dict["visual"];
        vp = this.core(view_dict);
        # print(nvp.norm(dim=1))
        return vp


class neko_structural_visual_only_interprinter(nn.Module):
    def __init__(this,feature_cnt,core=None):
        super(neko_structural_visual_only_interprinter,this).__init__();
        if core is None:
            this.core=resnet18np(outch=feature_cnt);
        else:
            this.core=core;
    def forward(this,view_dict) :
        # vis_proto=view_dict["visual"];
        vp=this.core(view_dict);
        return vp.view(vp.shape[0],-1);
        # print(nvp.norm(dim=1))

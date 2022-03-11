import torch.nn as nn
from neko_sdk.AOF.blocks import BasicBlockNoLens
from neko_sdk.AOF.neko_std import neko_spatial_transform_deform_conv_3x3,neko_spatial_transform_deform_conv_3x3_lr;

class neko_std_wrapper(nn.Module):
    def __init__(this,ifc,ofc):
        super(neko_std_wrapper, this).__init__();
        this.core=neko_spatial_transform_deform_conv_3x3(ifc,ofc);
    def forward(this,x):
        return this.core(x),None;

class neko_stdlr_wrapper(nn.Module):
    def __init__(this,ifc,ofc):
        super(neko_stdlr_wrapper, this).__init__();
        this.core=neko_spatial_transform_deform_conv_3x3_lr(ifc,ofc);
    def forward(this,x):
        return this.core(x),None;

class neko_reslayer_std(nn.Module):
    def __init__(this,in_planes, planes, blocks=1, stride=1):
        super(neko_reslayer_std, this).__init__()
        this.in_planes=in_planes
        this.downsample = None
        if stride != 1 or in_planes != planes * BasicBlockNoLens.expansion:
            this.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes * BasicBlockNoLens.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * BasicBlockNoLens.expansion),
            )

        this.layers = []
        this.layers.append(BasicBlockNoLens(this.in_planes, planes, stride, this.downsample))
        this.add_module("blk" + "init", this.layers[-1]);
        in_planes = planes * BasicBlockNoLens.expansion
        this.layers.append(neko_std_wrapper(in_planes, in_planes))
        this.add_module("STDL" , this.layers[-1]);
        for i in range(1, blocks):
            this.layers.append(BasicBlockNoLens(in_planes, planes))
            this.add_module("blk"+str(i),this.layers[-1]);
        this.out_planes=planes;

    def forward(this, input):
        fields=[];
        feat=input;
        for l in  this.layers:
            feat,f=l(feat);
            if(f is not None):
                fields.append(f);
        return feat,fields;

class neko_reslayer_stdlr(nn.Module):
    def __init__(this,in_planes, planes, blocks=1, stride=1):
        super(neko_reslayer_stdlr, this).__init__()
        this.in_planes=in_planes
        this.downsample = None
        if stride != 1 or in_planes != planes * BasicBlockNoLens.expansion:
            this.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes * BasicBlockNoLens.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * BasicBlockNoLens.expansion),
            )

        this.layers = []
        this.layers.append(BasicBlockNoLens(this.in_planes, planes, stride, this.downsample))
        this.add_module("blk" + "init", this.layers[-1]);
        in_planes = planes * BasicBlockNoLens.expansion
        this.layers.append(neko_stdlr_wrapper(in_planes, in_planes))
        this.add_module("STDL" , this.layers[-1]);
        for i in range(1, blocks):
            this.layers.append(BasicBlockNoLens(in_planes, planes))
            this.add_module("blk"+str(i),this.layers[-1]);
        this.out_planes=planes;

    def forward(this, input):
        fields=[];
        feat=input;
        for l in  this.layers:
            feat,f=l(feat);
            if(f is not None):
                fields.append(f);
        return feat,fields;
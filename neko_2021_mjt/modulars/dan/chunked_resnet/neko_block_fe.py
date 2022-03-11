import torch.nn

from neko_sdk.encoders.ocr_networks.dan.dan_resnet import conv1x1,conv3x3
from neko_sdk.AOF.neko_lens import neko_lens
from torch import nn
def make_init_layer_cco_wo_bn(inpch,outplanes,stride):
    conv = nn.Conv2d(inpch, outplanes, kernel_size=3, stride=stride, padding=1,
                           bias=False)
    relu = nn.ReLU(inplace=True)
    return {"conv":conv,"relu":relu};

def make_init_layer_wo_bn(inpch,outplanes,stride):
    conv = nn.Conv2d(inpch, outplanes, kernel_size=3, stride=stride, padding=1,
                           bias=False)
    relu = nn.ReLU(inplace=True)
    return {"conv":conv,"relu":relu};

def make_init_layer_bn(outplanes):
    bn = nn.BatchNorm2d(outplanes)
    return {"bn":bn};
class init_layer:
    def __init__(this, layer_dict,bn_dict):
        this.conv=layer_dict["conv"];
        this.bn=bn_dict["bn"];
        this.relu=layer_dict["relu"];
    def __call__(this,x):
        return this.relu(this.bn(this.conv(x)));

def make_block_wo_bn(inplanes, outplanes, stride=1):
    return{
        "conv1" :conv1x1(inplanes, outplanes),
        "relu" : nn.ReLU(inplace=True),
        "conv2" : conv3x3(outplanes, outplanes, stride),
   }
def make_block_bn(outplanes):
    return{
        "bn1": nn.BatchNorm2d(outplanes),
        "bn2": nn.BatchNorm2d(outplanes),
    }
# assembles from dicts.
# Generally this module does onw the modules---which means we do not save or load via it
class BasicBlock_ass:
    def __init__(this, layer_dict,bn_dict):
        this.conv1=layer_dict["conv1"];
        this.relu=layer_dict["relu"];
        this.bn1=bn_dict["bn1"];
        this.conv2=layer_dict["conv2"];
        this.bn2 = bn_dict["bn2"];
        this.downsample =False;
        if("downsample" in layer_dict):
            this.sample_conv=layer_dict["downsample"]["conv"];
            this.sample_bn=bn_dict["downsample"]["bn"];
            this.downsample = True;


    def __call__(this,x ):
        residual = x
        out = this.conv1(x)
        out = this.bn1(out)
        out = this.relu(out)

        out = this.conv2(out)
        out = this.bn2(out)

        if this.downsample:
            residual = this.sample_conv(x);
            residual = this.sample_bn(residual);
        out += residual
        out = this.relu(out)

        return out


def make_dowsample_layer_bn( expansion, planes):
    return {"bn":nn.BatchNorm2d(planes * expansion)};
def make_dowsample_layer( inplanes, expansion, planes, stride=1):
    return {"conv":nn.Conv2d(inplanes, planes * expansion,
                      kernel_size=1, stride=stride, bias=False)};

# we only decouple at layer level---- Or it can get seriously messy.
def make_body_layer_wo_bn( inplanes,blocks, planes, expansion, stride=1):
    ret_weight={};
    ret_weight["blocks"]={};
    ret_weight["blocks"]["0"]=make_block_wo_bn(inplanes,planes,stride);
    if stride != 1 or inplanes != planes * expansion:
        ret_weight["blocks"]["0"]["downsample"]=make_dowsample_layer(inplanes,expansion, planes,stride)
    for i in range(1, blocks):
        ret_weight["blocks"][str(i)]=make_block_wo_bn(planes,planes)
    return ret_weight;

def make_body_layer_bn( inplanes,blocks, planes, expansion, stride=1):
    ret_weight={};
    ret_weight["blocks"]={};
    ret_weight["blocks"]["0"]=make_block_bn(planes);
    if stride != 1 or inplanes != planes * expansion:
        ret_weight["blocks"]["0"]["downsample"]=make_dowsample_layer_bn(expansion,planes)
    for i in range(1, blocks):
        ret_weight["blocks"][str(i)]=make_block_bn(planes)
    return ret_weight;

class dan_reslayer:
    def __init__(this, layer_dict,bn_dict):
        this.blocks=[];
        for k in layer_dict["blocks"]:
            blk=BasicBlock_ass(layer_dict["blocks"][k],bn_dict["blocks"][k])
            this.blocks.append(blk);
    def __call__(this,x):
        for l in this.blocks:
            x=l(x);
        return x;

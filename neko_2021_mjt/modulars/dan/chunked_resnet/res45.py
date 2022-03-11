import torch
from torch import nn
from neko_2021_mjt.modulars.dan.chunked_resnet.neko_block_fe import \
    make_init_layer_wo_bn,make_init_layer_bn,init_layer,\
    make_body_layer_wo_bn,make_body_layer_bn,dan_reslayer
from neko_sdk.AOF.neko_lens import neko_lens
# Dan config.
def res45tpt_wo_bn(inpch,oupch,strides,frac=1,ochs=None):
    retlayers={};
    blkcnt = [None, 3, 4, 6, 6, 3];
    if(ochs is None):
        ochs = [int(32*frac),int(32 * frac), int(64 * frac), int(128 * frac), int(256 * frac), oupch]
    retlayers["0"]=make_init_layer_wo_bn(inpch[0],ochs[0],strides[0]);
    retlayers["1"]=make_body_layer_wo_bn(ochs[0],blkcnt[1],ochs[1],1,strides[1]);
    retlayers["2"] = make_body_layer_wo_bn(ochs[1], blkcnt[2], ochs[2], 1, strides[2]);
    retlayers["3"] = make_body_layer_wo_bn(ochs[2], blkcnt[3], ochs[3], 1, strides[3]);
    retlayers["4"] = make_body_layer_wo_bn(ochs[3], blkcnt[4], ochs[4], 1, strides[4]);
    retlayers["5"] = make_body_layer_wo_bn(ochs[4], blkcnt[5], ochs[5], 1, strides[5]);
    return retlayers;
def res45_wo_bn(inpch,oupch,strides,frac=1,ochs=None):
    retlayers={};
    blkcnt = [None, 3, 4, 6, 6, 3];
    if(ochs is None):
        ochs = [int(32*frac),int(32 * frac), int(64 * frac), int(128 * frac), int(256 * frac), oupch]
    retlayers["0"]=make_init_layer_wo_bn(inpch[0],ochs[0],strides[0]);
    retlayers["1"]=make_body_layer_wo_bn(ochs[0],blkcnt[1],ochs[1],1,strides[1]);
    retlayers["2"] = make_body_layer_wo_bn(ochs[1], blkcnt[2], ochs[2], 1, strides[2]);
    retlayers["3"] = make_body_layer_wo_bn(ochs[2], blkcnt[3], ochs[3], 1, strides[3]);
    retlayers["4"] = make_body_layer_wo_bn(ochs[3], blkcnt[4], ochs[4], 1, strides[4]);
    retlayers["5"] = make_body_layer_wo_bn(ochs[4], blkcnt[5], ochs[5], 1, strides[5]);
    return retlayers;
def res45_bn(inpch,oupch,strides,frac=1,ochs=None):
    blkcnt = [None, 3, 4, 6, 6, 3];
    if ochs is None:
        ochs = [int(32*frac),int(32 * frac), int(64 * frac), int(128 * frac), int(256 * frac), oupch]
    retlayers = {};
    retlayers["0"] = make_init_layer_bn(ochs[0]);
    retlayers["1"] = make_body_layer_bn(ochs[0], blkcnt[1], ochs[1], 1, strides[1]);
    retlayers["2"] = make_body_layer_bn(ochs[1], blkcnt[2], ochs[2], 1, strides[2]);
    retlayers["3"] = make_body_layer_bn(ochs[2], blkcnt[3], ochs[3], 1, strides[3]);
    retlayers["4"] = make_body_layer_bn(ochs[3], blkcnt[4], ochs[4], 1, strides[4]);
    retlayers["5"] = make_body_layer_bn(ochs[4], blkcnt[5], ochs[5], 1, strides[5]);
    return retlayers;
# OSOCR config. Seems they have much better perf due to the heavier layout
# The called the method ``pami'' www
def res45ptpt_wo_bn(inpch,oupch,strides,frac=1,ochs=None):
    retlayers={};
    blkcnt = [None, 3, 4, 6, 6, 3];
    if(ochs is None):
        ochs = [int(32*frac),int(32 * frac), int(64 * frac), int(128 * frac), int(256 * frac), oupch]
    retlayers["0"]=make_init_layer_wo_bn(inpch[0],ochs[0],strides[0]);
    retlayers["1"]=make_body_layer_wo_bn(ochs[0],blkcnt[1],ochs[1],1,strides[1]);
    retlayers["2"] = make_body_layer_wo_bn(ochs[1], blkcnt[2], ochs[2], 1, strides[2]);
    retlayers["3"] = make_body_layer_wo_bn(ochs[2], blkcnt[3], ochs[3], 1, strides[3]);
    retlayers["4"] = make_body_layer_wo_bn(ochs[3], blkcnt[4], ochs[4], 1, strides[4]);
    retlayers["5"] = make_body_layer_wo_bn(ochs[4], blkcnt[5], ochs[5], 1, strides[5]);
    return retlayers;
def res45p_wo_bn(inpch,oupch,strides,frac=1,ochs=None):
    retlayers={};
    blkcnt = [None, 3, 4, 6, 6, 3];
    if(ochs is None):
        ochs = [int(32*frac),int(64 * frac), int(128 * frac), int(256 * frac), int(512 * frac), oupch]
    retlayers["0"]=make_init_layer_wo_bn(inpch[0],ochs[0],strides[0]);
    retlayers["1"]=make_body_layer_wo_bn(ochs[0],blkcnt[1],ochs[1],1,strides[1]);
    retlayers["2"] = make_body_layer_wo_bn(ochs[1], blkcnt[2], ochs[2], 1, strides[2]);
    retlayers["3"] = make_body_layer_wo_bn(ochs[2], blkcnt[3], ochs[3], 1, strides[3]);
    retlayers["4"] = make_body_layer_wo_bn(ochs[3], blkcnt[4], ochs[4], 1, strides[4]);
    retlayers["5"] = make_body_layer_wo_bn(ochs[4], blkcnt[5], ochs[5], 1, strides[5]);
    return retlayers;
def res45p_bn(inpch,oupch,strides,frac=1,ochs=None):
    blkcnt = [None, 3, 4, 6, 6, 3];
    if ochs is None:
        ochs = [int(32*frac),int(64 * frac), int(128 * frac), int(256 * frac), int(512 * frac), oupch]
    retlayers = {};
    retlayers["0"] = make_init_layer_bn(ochs[0]);
    retlayers["1"] = make_body_layer_bn(ochs[0], blkcnt[1], ochs[1], 1, strides[1]);
    retlayers["2"] = make_body_layer_bn(ochs[1], blkcnt[2], ochs[2], 1, strides[2]);
    retlayers["3"] = make_body_layer_bn(ochs[2], blkcnt[3], ochs[3], 1, strides[3]);
    retlayers["4"] = make_body_layer_bn(ochs[3], blkcnt[4], ochs[4], 1, strides[4]);
    retlayers["5"] = make_body_layer_bn(ochs[4], blkcnt[5], ochs[5], 1, strides[5]);
    return retlayers;

class res45_net:
    def __init__(this,layer_dict,bn_dict):
        this.init_layer=init_layer(layer_dict["0"],bn_dict["0"]);
        this.res_layer1 = dan_reslayer(layer_dict["1"], bn_dict["1"]);
        this.res_layer2 = dan_reslayer(layer_dict["2"], bn_dict["2"]);
        this.res_layer3 = dan_reslayer(layer_dict["3"], bn_dict["3"]);
        this.res_layer4 = dan_reslayer(layer_dict["4"], bn_dict["4"]);
        this.res_layer5 = dan_reslayer(layer_dict["5"], bn_dict["5"]);

    def __call__(this, x):
        ret=[];
        x=this.init_layer(x.contiguous());
        x = this.res_layer1(x);
        x = this.res_layer2(x);
        ret.append(x);
        x = this.res_layer3(x);
        x = this.res_layer4(x);
        ret.append(x);
        x = this.res_layer5(x);
        ret.append(x);
        return ret;

class res45_net_orig:
    def cuda(this):
        pass;
    def __init__(this,layer_dict,bn_dict):
        this.init_layer=init_layer(layer_dict["0"],bn_dict["0"]);
        this.res_layer1 = dan_reslayer(layer_dict["1"], bn_dict["1"]);
        this.res_layer2 = dan_reslayer(layer_dict["2"], bn_dict["2"]);
        this.res_layer3 = dan_reslayer(layer_dict["3"], bn_dict["3"]);
        this.res_layer4 = dan_reslayer(layer_dict["4"], bn_dict["4"]);
        this.res_layer5 = dan_reslayer(layer_dict["5"], bn_dict["5"]);

    def __call__(this, x):
        ret=[];
        x=this.init_layer(x.contiguous());
        tmp_shape = x.size()[2:]
        x = this.res_layer1(x);
        if x.size()[2:] != tmp_shape:
            tmp_shape = x.size()[2:]
            ret.append(x)
        x = this.res_layer2(x);
        if x.size()[2:] != tmp_shape:
            tmp_shape = x.size()[2:];
            ret.append(x);
        x = this.res_layer3(x);
        if x.size()[2:] != tmp_shape:
            tmp_shape = x.size()[2:];
            ret.append(x);
        x = this.res_layer4(x);
        if x.size()[2:] != tmp_shape:
            tmp_shape = x.size()[2:]
            ret.append(x);
        x = this.res_layer5(x);
        ret.append(x);
        return ret;

class res45_net_tpt:
    def cuda(this):
        pass;
    def __init__(this,layer_dict,bn_dict,lens):
        this.init_layer=init_layer(layer_dict["0"],bn_dict["0"]);
        this.lens=lens;
        this.res_layer1 = dan_reslayer(layer_dict["1"], bn_dict["1"]);
        this.res_layer2 = dan_reslayer(layer_dict["2"], bn_dict["2"]);
        this.res_layer3 = dan_reslayer(layer_dict["3"], bn_dict["3"]);
        this.res_layer4 = dan_reslayer(layer_dict["4"], bn_dict["4"]);
        this.res_layer5 = dan_reslayer(layer_dict["5"], bn_dict["5"]);

    def __call__(this, x):
        ret=[];
        x=this.init_layer(x.contiguous());
        x,_=this.lens(x);
        tmp_shape = x.size()[2:]
        x = this.res_layer1(x);
        if x.size()[2:] != tmp_shape:
            tmp_shape = x.size()[2:]
            ret.append(x)
        x = this.res_layer2(x);
        if x.size()[2:] != tmp_shape:
            tmp_shape = x.size()[2:];
            ret.append(x);
        x = this.res_layer3(x);
        if x.size()[2:] != tmp_shape:
            tmp_shape = x.size()[2:];
            ret.append(x);
        x = this.res_layer4(x);
        if x.size()[2:] != tmp_shape:
            tmp_shape = x.size()[2:]
            ret.append(x);
        x = this.res_layer5(x);
        ret.append(x);
        return ret;

class res45_net_ptpt:
    def cuda(this):
        pass;
    def __init__(this,layer_dict,bn_dict,lens):
        this.init_layer=init_layer(layer_dict["0"],bn_dict["0"]);
        this.lens=lens;
        this.res_layer1 = dan_reslayer(layer_dict["1"], bn_dict["1"]);
        this.res_layer2 = dan_reslayer(layer_dict["2"], bn_dict["2"]);
        this.res_layer3 = dan_reslayer(layer_dict["3"], bn_dict["3"]);
        this.res_layer4 = dan_reslayer(layer_dict["4"], bn_dict["4"]);
        this.res_layer5 = dan_reslayer(layer_dict["5"], bn_dict["5"]);

    def __call__(this, x):
        ret=[];
        x=this.init_layer(x.contiguous());
        x,_=this.lens(x);
        tmp_shape = x.size()[2:]
        x = this.res_layer1(x);
        if x.size()[2:] != tmp_shape:
            tmp_shape = x.size()[2:]
            ret.append(x)
        x = this.res_layer2(x);
        if x.size()[2:] != tmp_shape:
            tmp_shape = x.size()[2:];
            ret.append(x);
        x = this.res_layer3(x);
        if x.size()[2:] != tmp_shape:
            tmp_shape = x.size()[2:];
            ret.append(x);
        x = this.res_layer4(x);
        if x.size()[2:] != tmp_shape:
            tmp_shape = x.size()[2:]
            ret.append(x);
        x = this.res_layer5(x);
        ret.append(x);
        return ret;
# so this thing keeps the modules and
class neko_r45_binorm(nn.Module):
    def setup_modules(this,mdict,prefix):
        for k in mdict:
            if(type(mdict[k]) is dict):
                this.setup_modules(mdict[k],prefix+"_"+k);
            else:
                this.add_module(prefix+"_"+k,mdict[k]);
    def setup_bn_modules(this,mdict,prefix):
        for k in mdict:
            if (type(mdict[k]) is dict):
                this.setup_bn_modules(mdict[k], prefix + "_" + k);
            else:
                id=len(this.bns);
                if(prefix not in this.bn_dict):
                    this.bn_dict[prefix]=[];
                this.bn_dict[prefix].append(id);
                this.add_module(prefix + "_" + k, mdict[k]);
                this.bns.append(mdict[k])
    def freezebnprefix(this,prefix):
        for i in this.bn_dict[prefix]:
            this.bns[i].eval();

    def unfreezebnprefix(this, prefix):
        for i in this.bn_dict[prefix]:
            this.bns[i].train();

    def freezebn(this):
        for i in this.bns:
            i.eval();
    def unfreezebn(this):
        for i in this.bns:
            i.train();

    def __init__(this, strides, compress_layer, input_shape,bogo_names,bn_names,hardness=2,oupch=512,expf=1):
        super(neko_r45_binorm, this).__init__()
        this.bogo_modules={};
        this.bn_dict={};
        layers = res45_wo_bn(input_shape, oupch, [(1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1)], 1);
        this.setup_modules(layers, "shared_fe");
        this.bns=[];
        for i in range(len(bogo_names)):
            name=bogo_names[i];
            bn_name=bn_names[i];
            bns = res45_bn(input_shape, oupch, [(1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1)], 1);
            this.bogo_modules[name] = res45_net(layers, bns);
            this.setup_bn_modules(bns, bn_name);

    def forward(self, input,debug=False):
        # This won't work as this is just a holder
        exit(9)
    def Iwantshapes(self):
        pseudo_input = torch.rand(1, self.input_shape[0], self.input_shape[1], self.input_shape[2])
        features,grio = self.model(pseudo_input)
        return [feat.size()[1:] for feat in features]
class neko_r45_binorm_orig(nn.Module):
    def freezebnprefix(this, prefix):
        for i in this.named_bn_dicts[prefix]:
            this.bns[i].eval();

    def unfreezebnprefix(this, prefix):
        for i in this.named_bn_dicts[prefix]:
            this.bns[i].train();

    def setup_bn_modules(this,mdict,prefix,gprefix):
        if(gprefix not in this.named_bn_dicts):
            this.named_bn_dicts[gprefix]=[];

        for k in mdict:
            if (type(mdict[k]) is dict):
                this.setup_bn_modules(mdict[k], prefix + "_" + k,gprefix);
            else:
                this.add_module(prefix + "_" + k, mdict[k]);
                this.named_bn_dicts[gprefix].append(len(this.bns));
                this.bns.append(mdict[k])


    def freezebn(this):
        for i in this.bns:
            i.eval();
    def unfreezebn(this):
        for i in this.bns:
            i.train();

    def setup_modules(this,mdict,prefix):
        for k in mdict:
            if(type(mdict[k]) is dict):
                this.setup_modules(mdict[k],prefix+"_"+k);
            else:
                this.add_module(prefix+"_"+k,mdict[k]);

    def __init__(this, strides, compress_layer, input_shape,bogo_names,bn_names,hardness=2,oupch=512,expf=1):
        super(neko_r45_binorm_orig, this).__init__()
        this.bogo_modules={};
        layers = res45_wo_bn(input_shape, oupch, [(1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1)],frac=expf);
        this.setup_modules(layers,"shared_fe");
        this.bns=[];
        this.named_bn_dicts={};
        for i in range(len(bogo_names)):
            name = bogo_names[i];
            bn_name = bn_names[i];
            bns = res45_bn(input_shape, oupch, [(1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1)], frac=expf);
            this.bogo_modules[name] = res45_net_orig(layers, bns);
            this.setup_bn_modules(bns, bn_name,bn_name);

class neko_r45_binorm_tpt(nn.Module):
    def freezebnprefix(this, prefix):
        for i in this.named_bn_dicts[prefix]:
            this.bns[i].eval();

    def unfreezebnprefix(this, prefix):
        for i in this.named_bn_dicts[prefix]:
            this.bns[i].train();

    def setup_bn_modules(this,mdict,prefix,gprefix):
        if(gprefix not in this.named_bn_dicts):
            this.named_bn_dicts[gprefix]=[];

        for k in mdict:
            if (type(mdict[k]) is dict):
                this.setup_bn_modules(mdict[k], prefix + "_" + k,gprefix);
            else:
                this.add_module(prefix + "_" + k, mdict[k]);
                this.named_bn_dicts[gprefix].append(len(this.bns));
                this.bns.append(mdict[k])


    def freezebn(this):
        for i in this.bns:
            i.eval();
    def unfreezebn(this):
        for i in this.bns:
            i.train();

    def setup_modules(this,mdict,prefix):
        for k in mdict:
            if(type(mdict[k]) is dict):
                this.setup_modules(mdict[k],prefix+"_"+k);
            else:
                this.add_module(prefix+"_"+k,mdict[k]);

    def __init__(this, strides, compress_layer, input_shape,bogo_names,bn_names,hardness=2,oupch=512,expf=1):
        super(neko_r45_binorm_tpt, this).__init__()
        this.bogo_modules={};
        layers = res45_wo_bn(input_shape, oupch, [(1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1)],frac=expf);
        this.setup_modules(layers,"shared_fe");
        this.bns=[];
        this.tpt= neko_lens(int(32*expf),1,1,hardness);
        this.named_bn_dicts={};
        for i in range(len(bogo_names)):
            name = bogo_names[i];
            bn_name = bn_names[i];
            bns = res45_bn(input_shape, oupch, [(1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1)], frac=expf);
            this.bogo_modules[name] = res45_net_orig(layers, bns);
            this.setup_bn_modules(bns, bn_name,bn_name);

    def forward(self, input,debug=False):
        # This won't work as this is just a holder
        exit(9)
    def Iwantshapes(self):
        pseudo_input = torch.rand(1, self.input_shape[0], self.input_shape[1], self.input_shape[2])
        features,grio = self.model(pseudo_input)
        return [feat.size()[1:] for feat in features]
class neko_r45_binorm_ptpt(nn.Module):
    def freezebnprefix(this, prefix):
        for i in this.named_bn_dicts[prefix]:
            this.bns[i].eval();

    def unfreezebnprefix(this, prefix):
        for i in this.named_bn_dicts[prefix]:
            this.bns[i].train();

    def setup_bn_modules(this,mdict,prefix,gprefix):
        if(gprefix not in this.named_bn_dicts):
            this.named_bn_dicts[gprefix]=[];

        for k in mdict:
            if (type(mdict[k]) is dict):
                this.setup_bn_modules(mdict[k], prefix + "_" + k,gprefix);
            else:
                this.add_module(prefix + "_" + k, mdict[k]);
                this.named_bn_dicts[gprefix].append(len(this.bns));
                this.bns.append(mdict[k])


    def freezebn(this):
        for i in this.bns:
            i.eval();
    def unfreezebn(this):
        for i in this.bns:
            i.train();

    def setup_modules(this,mdict,prefix):
        for k in mdict:
            if(type(mdict[k]) is dict):
                this.setup_modules(mdict[k],prefix+"_"+k);
            else:
                this.add_module(prefix+"_"+k,mdict[k]);

    def __init__(this, strides, compress_layer, input_shape,bogo_names,bn_names,hardness=2,oupch=512,expf=1):
        super(neko_r45_binorm_ptpt, this).__init__()
        this.bogo_modules={};
        layers = res45p_wo_bn(input_shape, oupch, [(1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1)],frac=expf);
        this.setup_modules(layers,"shared_fe");
        this.bns=[];
        this.tpt= neko_lens(int(32*expf),1,1,hardness);
        this.named_bn_dicts={};
        for i in range(len(bogo_names)):
            name = bogo_names[i];
            bn_name = bn_names[i];
            bns = res45p_bn(input_shape, oupch, [(1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1)], frac=expf);
            this.bogo_modules[name] = res45_net_orig(layers, bns);
            this.setup_bn_modules(bns, bn_name,bn_name);

    def forward(self, input,debug=False):
        # This won't work as this is just a holder
        exit(9)
    def Iwantshapes(self):
        pseudo_input = torch.rand(1, self.input_shape[0], self.input_shape[1], self.input_shape[2])
        features,grio = self.model(pseudo_input)
        return [feat.size()[1:] for feat in features]

class neko_r45_binorm_heavy_head(nn.Module):
    def setup_modules(this,mdict,prefix):
        for k in mdict:
            if(type(mdict[k]) is dict):
                this.setup_modules(mdict[k],prefix+"_"+k);
            else:
                this.add_module(prefix+"_"+k,mdict[k]);

    def __init__(this, strides, compress_layer, input_shape,bogo_names,bn_names,hardness=2,oupch=512,expf=1):
        super(neko_r45_binorm_heavy_head, this).__init__()
        this.bogo_modules={};
        ochs = [int(64*expf),int(64 * expf), int(64 * expf), int(128 * expf), int(256 * expf), oupch]

        layers = res45_wo_bn(input_shape, oupch, [(1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1)], 1,ochs=ochs);
        this.setup_modules(layers,"shared_fe");
        for i in range(len(bogo_names)):
            name = bogo_names[i];
            bn_name = bn_names[i];
            bns = res45_bn(input_shape, oupch, [(1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1)], 1);
            this.bogo_modules[name] = res45_net_orig(layers, bns);
            this.setup_modules(bns, bn_name);
       

    def forward(self, input,debug=False):
        # This won't work as this is just a holder
        exit(9)
    def Iwantshapes(self):
        pseudo_input = torch.rand(1, self.input_shape[0], self.input_shape[1], self.input_shape[2])
        features,grio = self.model(pseudo_input)
        return [feat.size()[1:] for feat in features]


if __name__ == '__main__':
    layers=res45_wo_bn(3,512,[(1,1), (2,2), (1,1), (2,2), (1,1), (1,1)],1);
    bns=res45_bn(3,512,[(1,1), (2,2), (1,1), (2,2), (1,1), (1,1)],1);
    a=res45_net(layers,bns);
    t=torch.rand([1,3,32,128]);
    r=a(t);
    pass;

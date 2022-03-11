import torch
from torch import nn
from torch.nn import functional as trnf
import random;
from neko_sdk.ocr_modules.neko_interprinter import neko_visual_only_interprinter,neko_visual_only_interprinterR34;
import numpy as np;
from neko_sdk.ocr_modules.neko_prototyper_gen2.neko_abstractract_sampler import neko_prototype_sampler_static

import regex

class neko_prototyper(nn.Module):
    PROTOENGINE=neko_visual_only_interprinter;
    def __init__(this,output_channel,spks,dropout=None,capacity=512):
        super(neko_prototyper,this).__init__()
        this.output_channel=output_channel;
        this.sp_cnt=len(spks);
        this.proto_engine = this.PROTOENGINE(this.output_channel);
        this.dev_ind = torch.nn.Parameter(torch.rand([1]));
        this.EOS=0
        this.sp_protos = torch.nn.Parameter(torch.rand([
            this.sp_cnt, this.output_channel]).float() * 2 - 1);
        this.register_parameter("sp_proto", this.sp_protos);
        if (dropout is not None):
            this.drop = torch.nn.Dropout(p=0.3);
        else:
            this.drop = None;
        print("DEBUG-SDFGASDFGSDGASFGSD",dropout);
        # split if too many;
        this.capacity=capacity;
        this.freeze_bn_affine=False;

    def freezebn(this):
        for m in this.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                if this.freeze_bn_affine:
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False

    def unfreezebn(this):
        for m in this.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.train();
                if this.freeze_bn_affine:
                    m.weight.requires_grad = True
                    m.bias.requires_grad = True

    def forward(this,normprotos,rot=0,use_sp=True):
        if(len(normprotos)<=this.capacity):
            # pimage=torch.cat(normprotos).to(this.dev_ind.device);
            pimage=torch.cat(normprotos).contiguous().to(this.dev_ind.device);

            if(rot>0):
                pimage=torch.rot90(pimage,rot,[2,3]);

            if(pimage.shape[1]==1):
                pimage=pimage.repeat([1,3,1,1]);
            if (use_sp):
                proto = [this.sp_protos,this.proto_engine(pimage)];
            else:
                proto = [this.proto_engine(pimage)];
        else:
            if (use_sp):
                proto = [this.sp_protos];
            else:
                proto = [];
            for s in range(0, len(normprotos),this.capacity):
                pimage = torch.cat(normprotos[s:s+this.capacity]).contiguous().to(this.dev_ind.device);
                if (rot > 0):
                    pimage = torch.rot90(pimage, rot,[2,3]);
                if (pimage.shape[1] == 1):
                    pimage = pimage.repeat([1, 3, 1, 1]);
                    proto.append(this.proto_engine(pimage))

        allproto = trnf.normalize(torch.cat(proto),dim=1,eps=0.0009);
        if (this.drop):
            allproto = this.drop(allproto);
        pass;
        return allproto.contiguous();
class neko_prototyperR34(neko_prototyper):
    PROTOENGINE=neko_visual_only_interprinterR34;


class neko_prototype_sampler_basic(neko_prototype_sampler_static):
    def train(this,training=True):
        pass;
    def eval(this):
        pass;
    def cuda(this):
        pass;

    # defines sampler
    def setup_sampler(this,sampler_args):
        if sampler_args is None:
            max_match_size=512;
            val_frac=0.8;
            neg_servant=True;
        else:
            max_match_size = sampler_args["max_batch_size"];
            val_frac=sampler_args["val_frac"];
            neg_servant=sampler_args["neg_servant"];
        this.max_batch_size=max_match_size;
        this.val_frac=val_frac;
        this.neg_servant=neg_servant;


    def debug(this,normpids,labels):
        normprotos = [this.norm_protos[i - this.sp_cnt] for i in normpids];
        protos=((torch.cat(normprotos, dim=-1).squeeze(0).squeeze(0)+1)*127.5).detach().cpu().numpy().astype(np.uint8);
        import cv2;
        cv2.imshow(labels,protos[:,:32*32])
        cv2.waitKey(0);

    def grab_cluster(this,ch):
        chid=this.label_dict[ch];
        ret={chid};
        if this.masters_share:
            ret.add(this.masters[chid]);
            ret=ret.union(this.servants[this.masters[chid]]);
        return ret;
    def dump_all_impl(this,use_sp=True):
        if(use_sp):
            trsps = [this.EOS];
        else:
            trsps=[];
        trchs = list(set([this.label_dict[i] for i in this.shaped_characters]));
        normprotos = [this.norm_protos[i - this.sp_cnt] for i in trchs];
        plabels, tdicts = this.get_plabel_and_dict(trsps, trchs)
        return normprotos, plabels, tdicts;
    def get_sampled_ids(this,plain_chars_in_data):
        cntval = int(len(plain_chars_in_data) * this.val_frac);
        cntval = min(this.max_batch_size - this.sp_cnt, cntval);
        trchs=set();
        related_chars_in_data=set();
        random.shuffle(plain_chars_in_data);
        # make sure no missing centers--
        # or it may enforce "A" to look like "a" encoded by proto CNN
        remaining = cntval;
        for ch in plain_chars_in_data:
            if(ch not in this.label_dict):
                continue;
            new=this.grab_cluster(ch);
            ns=trchs.union(new);
            related_chars_in_data=related_chars_in_data.union(new);
            delta=len(ns)-len(trchs);
            if(delta<=remaining):
                trchs=ns;
                remaining-=delta;
        remaining=this.max_batch_size-this.sp_cnt-len(trchs);
        plain_charid_not_in_data=list(this.shaped_ids-related_chars_in_data);
        random.shuffle(plain_charid_not_in_data);
        for chid in plain_charid_not_in_data:
            if chid not in trchs:
                if (remaining == 0):
                    break;
                if (this.neg_servant==False and this.masters[chid]!=chid):
                    continue;
                remaining-=1;
                trchs.add(chid);

        trsps=set([this.label_dict[i] for i in this.sp_tokens]);
        return trsps,trchs;

    def sample_charset_by_text(this,text_batch,use_sp=True):
        plain_chars_in_data = this.get_occured(text_batch)
        trsps,trchs=this.get_sampled_ids(plain_chars_in_data);
        trchs=list(trchs);
        if(use_sp is not False):
            trsps=list(trsps);
        else:
            trsps=[];
        plabels,tdicts=this.get_plabel_and_dict(trsps,trchs)
        normprotos=[this.norm_protos[i-this.sp_cnt] for i in trchs];
        # this.debug(trchs,"meow");
        return normprotos,plabels,tdicts;
    def sample_charset_by_textg(this,text_batch,use_sp=True):
        plain_chars_in_data = this.get_occured(text_batch)
        trsps,trchs=this.get_sampled_ids(plain_chars_in_data);
        trchs=list(trchs);
        if(use_sp is not False):
            trsps=list(trsps);
        else:
            trsps=[];
        plabels,tdicts,gtdicts=this.get_plabel_and_dictg(trsps,trchs)
        normprotos=[this.norm_protos[i-this.sp_cnt] for i in trchs];
        # this.debug(trchs,"meow");
        return normprotos,plabels,tdicts,gtdicts;
    def sample_charset_by_text2(this,text_batch):
        plain_chars_in_data = this.get_occured(text_batch)
        trsps,trchs=this.get_sampled_ids(plain_chars_in_data);
        trchs=list(trchs);
        trsps=list(trsps);
        plabels,tdicts=this.get_plabel_and_dict(trsps,trchs)
        normprotos=[this.norm_protos[i-this.sp_cnt] for i in trchs];
        # this.debug(trchs,"meow");
        return normprotos,plabels,tdicts;

    def sample_charset_by_text_both(this,text_batch):
        b="";
        for _ in text_batch: b+=_;

        plain_chars_in_data=this.get_occured(text_batch);
        trsps,trchs=this.get_sampled_ids(plain_chars_in_data);
        trchs=list(trchs);
        trsps=list(trsps);
        normprotos=[this.norm_protos[i-this.sp_cnt] for i in trchs];
        plabels_cased,tdicts_cased=this.get_plabel_and_dict_core(trsps,trchs,False)
        plabels_uncased, tdicts_uncased = this.get_plabel_and_dict_core(trsps, trchs,True);
        # this.debug(trchs,"meow");
        return normprotos,[plabels_uncased,plabels_cased],[tdicts_uncased,tdicts_cased];
class neko_prototype_sampler_fsl(neko_prototype_sampler_basic):
    def split(this, s):
        # not hurt if we have one more meme. In fact we need a random two-character non-sense
        return s.split("⑤⑨");
    def get_occured(this, text_batch):
        b = [];
        for _ in text_batch: b += _.split("⑤⑨");
        return b;



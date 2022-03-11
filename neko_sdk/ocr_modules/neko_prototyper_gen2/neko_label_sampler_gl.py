import torch
import random;
import numpy as np;
from neko_sdk.ocr_modules.neko_prototyper_gen2.neko_abstractract_sampler import neko_prototype_sampler_static
class neko_prototype_sampler_gl(neko_prototype_sampler_static):
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
    def dump_all_impl(this, use_sp=True):
        if(use_sp):
            trsps = [this.label_dict[this.EOS]];
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
        plabels,gplabels,tdicts,gtdicts=this.get_plabel_and_dictg(trsps,trchs)
        normprotos=[this.norm_protos[i-this.sp_cnt] for i in trchs];
        # this.debug(trchs,"meow");
        return normprotos,plabels,gplabels,tdicts,gtdicts;

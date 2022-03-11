
# using mk5 DTD
import torch.nn.functional

from neko_2021_mjt.routines.neko_abstract_routines import neko_abstract_routine,neko_abstract_eval_routine;
from neko_2020nocr.dan.common.common import display_cfgs,load_dataset,Zero_Grad,Train_or_Eval,generate_optimizer,\
    Updata_Parameters,flatten_label,flatten_label_oc
from neko_2020nocr.dan.utils import Loss_counter,neko_os_Attention_AR_counter,neko_oswr_Attention_AR_counter
import numpy as np
import cv2;
from neko_2021_mjt.modulars.neko_inflater import neko_inflater
from neko_2021_mjt.routines.ocr_routines.mk5.osdan_routine_mk5 import neko_HDOS2C_routine_CFmk5
from neko_2021_mjt.routines.subroutines.context_subroutines.neko_ctx_subroutine import neko_ctx_subroutine_v1
from neko_2021_mjt.routines.subroutines.proto_making.contextual_v1 import mk_proto_contextual_v1
from neko_2021_mjt.routines.subroutines.fe_seq.TA import temporal_attention_v2,temporal_attention_v2dt
from neko_2021_mjt.routines.subroutines.mk8common import mk8_log,mklabel_mk8

# mk5 CF branch dropped predict-sample-predict support.
# A GP branch will be added if it's ever to be supported
# Mk7 CF branch uses CAM to perform length prediction, [s] is no more needed
# Let's re-introduce context.
def debug_mk8(chouts,ctxouts,gt):
    for i in range(len(chouts)):
        feat=None;
        if(chouts[i] !=ctxouts[i] and ctxouts[i]==gt[i]):
            feat = "A"
        elif(chouts[i]!=ctxouts[i] and chouts[i]==gt[i]):
            feat="B"
        elif(chouts[i]!=ctxouts[i]):
            feat="C"
        else:
            feat="D"
        if(feat is not None):
            print("["+feat+"]",gt[i],":",chouts[i],"->-",ctxouts[i]);

class neko_HDOS2C_routine_CFmk8(neko_HDOS2C_routine_CFmk5):
    def set_etc(this, args):
        this.maxT = args["maxT"];
        this.inflater = neko_inflater();
        this.PROTO_FN=mk_proto_contextual_v1;
        this.context_mod=neko_ctx_subroutine_v1();
        this.attf=temporal_attention_v2;
    def set_loggers(this,log_path,log_each,name):
        this.logger_dict = {
            "accr": neko_os_Attention_AR_counter("[" + name + "]" + "train_accr", False),
            "ctxaccr":neko_os_Attention_AR_counter("[" + name + "]" + "train_accr_ctx", False),
            "loss": Loss_counter("[" + name + "]" + "train_accr"),
        };
    def fp_impl(this, input_dict, module_dict,logger_dict,nEpoch,batch_idx):
        label=input_dict["label"];
        clips=input_dict["image"];
        preds=module_dict["preds"];
        if ("vdbg" not in input_dict):
            DBGKEY = None;
        else:
            DBGKEY = this.name;

        proto, fsp,csp, plabel,gplabel, tdict,gtdict = this.PROTO_FN(label,module_dict);
        label_flatten, glabel_flatten, target, gtarget, length, lengthl,idx = \
            mklabel_mk8(module_dict, proto, gtdict, plabel, tdict, label)
        target,gtarget, label_flatten,glabel_flatten,culength = \
            target.cuda(),gtarget.cuda(), label_flatten.cuda(),glabel_flatten.cuda(),length.cuda().long()

        out_emb,A,pred_length=this.attf(clips.cuda(),module_dict,length)
        # net forward
        # Length dring training is known.
        fout_emb,_=this.inflater.inflate(out_emb,length)


        lloss=torch.nn.functional.cross_entropy(pred_length,culength);
        logits=preds[0](fout_emb, proto,plabel);
        choutput, prdt_prob = module_dict["sampler"].model.decode(logits, length, proto, plabel, tdict);
        ctxloss,ctxout,cctxlogits=this.context_mod.engage(module_dict,fsp,csp,logits,lengthl,length,plabel,glabel_flatten,gtdict)

        vloss, terms_ = module_dict["losses"][0](proto, logits, label_flatten);

        tloss = vloss + lloss+ctxloss*0.5;
        terms={
            "total":tloss.item(),
            "length":lloss.item(),
            "visual":vloss.item(),
            "ctx":ctxloss.item(),
        }
        mk8_log(logger_dict,length,label,tdict,gtdict,target,gtarget,choutput,ctxout,terms,tloss.item(),DBGKEY)
        return tloss;


class neko_HDOS2C_routine_CFmk8a(neko_HDOS2C_routine_CFmk8):
    def fp_impl(this, input_dict, module_dict,logger_dict,nEpoch,batch_idx):
        label=input_dict["label"];
        clips=input_dict["image"];
        preds=module_dict["preds"];
        if ("vdbg" not in input_dict):
            DBGKEY = None;
        else:
            DBGKEY = this.name;

        proto, fsp,csp, plabel,gplabel, tdict,gtdict = this.PROTO_FN(label,module_dict);
        label_flatten, glabel_flatten, target, gtarget, length, lengthl,idx = \
            mklabel_mk8(module_dict, proto, gtdict, plabel, tdict, label)
        target,gtarget, label_flatten,glabel_flatten,culength = \
            target.cuda(),gtarget.cuda(), label_flatten.cuda(),glabel_flatten.cuda(),length.cuda().long()

        out_emb,A,pred_length=this.attf(clips.cuda(),module_dict,length)
        # net forward
        # Length dring training is known.
        fout_emb,_=this.inflater.inflate(out_emb,length)


        lloss=torch.nn.functional.cross_entropy(pred_length,culength);
        logits=preds[0](fout_emb, proto,plabel);
        choutput, prdt_prob = module_dict["sampler"].model.decode(logits, length, proto, plabel, tdict);
        ctxloss,ctxout,cctxlogits=this.context_mod.engage(module_dict,fsp,csp,logits,lengthl,length,plabel,glabel_flatten,gtdict)

        vloss, terms_ = module_dict["losses"][0](proto, logits, label_flatten);

        tloss = vloss + lloss+ctxloss;
        terms={
            "total":tloss.item(),
            "length":lloss.item(),
            "visual":vloss.item(),
            "ctx":ctxloss.item(),
        }
        mk8_log(logger_dict,length,label,tdict,gtdict,target,gtarget,choutput,ctxout,terms,tloss.item(),DBGKEY)
        return tloss;
class neko_HDOS2C_routine_CFmk8adt(neko_HDOS2C_routine_CFmk8a):
    def set_etc(this, args):
        this.maxT = args["maxT"];
        this.inflater = neko_inflater();
        this.PROTO_FN=mk_proto_contextual_v1;
        this.context_mod=neko_ctx_subroutine_v1();
        this.attf=temporal_attention_v2dt;


class neko_HDOS2C_eval_routine_CFmk8(neko_abstract_eval_routine):
    def pretest_impl(this,modular_dict,metaargs,**kwargs):
        rot = kwargs["rot"];
        normproto, plabels,gplabels, tdict,gbidict= modular_dict["sampler"].model.dump_allg(metaargs=metaargs,use_sp=False);
        fsp = modular_dict["semantic_branch"]()[:gbidict["[UNK]"] + 1];
        csp = modular_dict["semantic_branch"](gplabels);
        if (not rot):
            proto = modular_dict["prototyper"](normproto,use_sp=False);
        else:
            proto = modular_dict["prototyper"](normproto, rot);
        return {"proto":proto,"fsp":fsp,"csp":csp,"plabel":plabels,"tdict":tdict,"gtdict":gbidict};

        # tmetastart = time.time();

    def set_etc(this, args):
        this.maxT = args["maxT"];
        # if we know the context is meant to change, we refuse to use the context.
        # Dynamically ensemble will be introduced in MK9.
        # Where each stage will guess how reliable the result is---this, perhaps will overfit.
        if(args["mod_cvt_dicts"]["ctxmodule"]!="NEPnoneNEP"):
            this.has_ctx=True;
        else:
            this.has_ctx=False;
        this.inflater=neko_inflater();


    def set_loggers(this,log_path,name,args):

        try:
            if (args["measure_rej"] == True):
                this.logger_dict = {"accr": neko_oswr_Attention_AR_counter("[" + name + "]" + "test_accr", False),
                                    }
            else:
                this.logger_dict = {
                    "accr": neko_os_Attention_AR_counter("[" + name + "]" + "test_accr", False),
                    "loss": Loss_counter("[" + name + "]" + "train_accr"),
                };
        except:
            this.logger_dict = {
                "accr": neko_os_Attention_AR_counter("[" + name + "]" + "test_accr", False),
                "loss": Loss_counter("[" + name + "]" + "train_accr"),
            };
        if(this.has_ctx):
            this.logger_dict["ctxaccr"]= neko_os_Attention_AR_counter("[" + name + "]" + "test_accr_ctx", False);

    def test_impl(this,data_dict, module_dict,logger_dict):

        data,label,proto, plabel, tdict= \
        data_dict["image"],data_dict["label"],data_dict["proto"],data_dict["plabel"],data_dict["tdict"];
        preds=module_dict["preds"];
        seq=module_dict["seq"];
        sampler=module_dict["sampler"];


        data=data.cuda();
        features = module_dict["feature_extractor"](data)
        A,pred_length = module_dict["TA"](features)
        pred_length=pred_length.argmax(dim=-1)
        # A0=A.detach().clone();
        out_emb =seq(features[-1],A,None);
        # lossess = []
        beams_ = [];
        cbeams_=[];
        probs = [];
        # terms = [];

        loss = 0;
        nT,nB=out_emb.shape[0],out_emb.shape[1];
        logits = preds[0](out_emb.reshape([nT*nB,-1]), proto,plabel).reshape([nT,nB,-1]);
        logits,_ = this.inflater.inflate(logits, pred_length);
        if(module_dict["ctxmodule"] is not None):
            ctx_logits,compact_ctx_logits = module_dict["ctxmodule"](logits, list(pred_length.detach().cpu().numpy()), data_dict["csp"],
                                                  data_dict["fsp"], pred_length.max().item())
            d = data_dict["gtdict"];
            ctx_logits = ctx_logits[:, 1:] + logits;
            ctx_choutput, ctx_prdt_prob = sampler.model.decode(ctx_logits, pred_length, None, None, tdict);
            #ctx_choutput, ctx_prdt_prob = sampler.model.decode(ctx_logits, pred_length, None, None, d);
            cbeams_.append(ctx_choutput);
            logger_dict["ctxaccr"].add_iter(ctx_choutput, pred_length, label);

        choutput, prdt_prob = sampler.model.decode(logits, pred_length, proto, plabel, tdict);
        beams_.append(choutput);
        probs.append(prdt_prob);
        # loss_, terms_ = module_dict["losses"][i](proto, preds, label_flatten);
        # loss = loss_ + loss;
        # beams_.append(choutput);
            # terms.append(terms_);
        beams=[];
        for i in range(features[-1].shape[0]):
            beam = [];
            for j in range(len(beams_)):
                beam.append(beams_[j][i]);
            beams.append(beam)
        # A=A.max(dim=2)[0];
        if(label is not None):
            labelwunk=["".join([ch if ch in tdict else "⑨" for ch in w]) for w in label];
            logger_dict["accr"].add_iter(beams_[0],pred_length, labelwunk)
            if("vdbg"in data_dict and module_dict["ctxmodule"] is not None and label is not None):
                debug_mk8(choutput,ctx_choutput,label);
            if(module_dict["ctxmodule"] is not None):
                for i in range(len(beams)):
                    beams[i].append(cbeams_[0][i])
        return beams_[0],(probs[0],A.max(1)[0].detach()),beams;

    def vis_logits_impl(this,img,data_dict,module_dict,at_time):

        _, label, proto, plabel, tdict = \
            data_dict["image"], data_dict["label"], data_dict["proto"], data_dict["plabel"], data_dict["tdict"];
        data=img
        preds = module_dict["preds"];
        seq = module_dict["seq"];
        sampler = module_dict["sampler"];

        data = data.cuda();
        data=torch.nn.Parameter(data,requires_grad=True);
        features = module_dict["feature_extractor"](data)
        A, pred_length = module_dict["TA"](features)
        pred_length = pred_length.argmax(dim=-1)
        # A0=A.detach().clone();
        out_emb = seq(features[-1], A, None);
        # lossess = []
        beams_ = [];
        cbeams_ = [];
        probs = [];
        # terms = [];

        loss = 0;
        nT, nB = out_emb.shape[0], out_emb.shape[1];
        logits = preds[0](out_emb.reshape([nT * nB, -1]), proto, plabel).reshape([nT, nB, -1]);
        logits, _ = this.inflater.inflate(logits, pred_length);
        if (len(logits) <= at_time):
            return None;
        return logits[at_time]
        # A.detach().reshape(A.shape[0], A.shape[1], A.shape[2], A.shape[3]).sum(2)




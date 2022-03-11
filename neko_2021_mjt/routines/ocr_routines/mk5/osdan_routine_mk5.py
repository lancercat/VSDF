# using mk5 DTD
from neko_2021_mjt.routines.neko_abstract_routines import neko_abstract_routine,neko_abstract_eval_routine;
from neko_2020nocr.dan.common.common import display_cfgs,load_dataset,Zero_Grad,Train_or_Eval,generate_optimizer,\
    Updata_Parameters,flatten_label,flatten_label_oc
from neko_2020nocr.dan.utils import Loss_counter,neko_os_Attention_AR_counter
import numpy as np
import cv2;
from neko_2021_mjt.modulars.neko_inflater import neko_inflater
# mk5 CF branch dropped predict-sample-predict support.
# A GP branch will be added if it's ever to be supported
class neko_HDOS2C_routine_CFmk5(neko_abstract_routine):
    def mk_proto(this,label,sampler,prototyper):
        normprotos, plabel, tdict=sampler.model.sample_charset_by_text(label)
        # im=(torch.cat(normprotos,3)*127+128)[0][0].numpy().astype(np.uint8);
        # cv2.imshow("alphabets",im);
        # print([tdict[label.item()] for label in plabel]);
        # cv2.waitKey(0);
        proto=prototyper(normprotos);

        semb=None
        return proto, semb, plabel, tdict
    def set_etc(this, args):
        this.maxT = args["maxT"];
        this.inflater=neko_inflater();

    def set_loggers(this,log_path,log_each,name):
        this.logger_dict={
            "accr": neko_os_Attention_AR_counter("["+name+"]"+"train_accr",False),
            "loss": Loss_counter("[" + name + "]" + "train_accr"),
        };
    def show_clip(this,clip,label):
        im=(clip.detach()*255).permute(0,2,3,1).cpu().numpy().astype(np.uint8);
        for i in range(len(im)):
            cv2.imshow(label[i],im[i]);
        cv2.waitKey(0);
    def fe_seq(this,clips,module_dict,length):
        seq=module_dict["seq"];
        features = module_dict["feature_extractor"](clips.cuda())
        features=[f.contiguous() for f in features];

        A = module_dict["CAM"](features)
        out_emb = seq(features[-1], A, length);
        return out_emb,A;

    def fp_impl(this, input_dict, module_dict,logger_dict,nEpoch,batch_idx):
        label=input_dict["label"];
        clips=input_dict["image"];
        prototyper=module_dict["prototyper"]
        sampler=module_dict["sampler"];
        preds=module_dict["preds"];

        proto, semb, plabel, tdict = this.mk_proto(label,sampler,prototyper);
        target = sampler.model.encode(proto, plabel, tdict, label);
        label_flatten, length = flatten_label(target)
        target, label_flatten = target.cuda(), label_flatten.cuda()

        out_emb,A=this.fe_seq(clips,module_dict,length)
        # net forward
        # Length dring training is known.
        fout_emb,_=this.inflater.inflate(out_emb,length)
        lossess=[]
        beams=[];
        probs=[];
        terms=[];
        loss=0;
        for i in range(len(preds)):
            logits=preds[i](fout_emb, proto,plabel);
            choutput, prdt_prob = sampler.model.decode(logits, length, proto, plabel, tdict);
            loss_, terms_ = module_dict["losses"][i](proto, logits, label_flatten);
            loss=loss_+loss;
            beams.append(choutput);
            terms.append(terms_);

        # computing accuracy and loss
        tarswunk = ["".join([tdict[i.item()] for i in target[j]]).replace('[s]', "") for j in
                    range(len(target))];
        logger_dict["accr"].add_iter(beams[0], length, tarswunk)
        logger_dict["loss"].add_iter(loss, terms[0])
        return loss;

class neko_HDOS2C_eval_routine_CFmk5(neko_abstract_eval_routine):
    def set_etc(this, args):
        this.maxT = args["maxT"];
        this.inflater=neko_inflater();

    def set_loggers(this,log_path,name,args):
        this.logger_dict={
            "accr": neko_os_Attention_AR_counter("[" + name + "]" + "test_accr", False),
            "loss": Loss_counter("[" + name + "]" + "train_accr"),
        };
    def test_impl(this,data_dict, module_dict,logger_dict):
        data,label,proto, plabel, tdict= \
        data_dict["image"],data_dict["label"],data_dict["proto"],data_dict["plabel"],data_dict["tdict"];
        preds=module_dict["preds"];
        seq=module_dict["seq"];
        sampler=module_dict["sampler"];


        data=data.cuda();
        features = module_dict["feature_extractor"](data)
        A = module_dict["CAM"](features)

        # A0=A.detach().clone();
        out_emb =seq(features[-1],A,None);
        # lossess = []
        beams_ = [];
        probs = [];
        # terms = [];

        loss = 0;
        for i in range(len(preds)):
            nT,nB=out_emb.shape[0],out_emb.shape[1];
            logits = preds[i](out_emb.reshape([nT*nB,-1]), proto,plabel).reshape([nT,nB,-1]);
            logits, length = this.inflater.inflate(logits, None);

            choutput, prdt_prob = sampler.model.decode(logits, length, proto, plabel, tdict);
            beams_.append(choutput);
            probs.append(prdt_prob);
            # loss_, terms_ = module_dict["losses"][i](proto, preds, label_flatten);
            # loss = loss_ + loss;
            beams_.append(choutput);
            # terms.append(terms_);
        beams=[];
        for i in range(features[-1].shape[0]):
            beam = [];
            for j in range(len(beams_)):
                beam.append(beams_[j][i]);
            beams.append(beam)
        # A=A.max(dim=2)[0];
        logger_dict["accr"].add_iter(beams_[0],length, label)
        return beams_[0],(probs[0],A.max(1)[0].detach()),beams;

        # A.detach().reshape(A.shape[0], A.shape[1], A.shape[2], A.shape[3]).sum(2)

    def pretest_impl(this,modular_dict,metaargs,**kwargs):
        rot=kwargs["rot"];
        normproto, plabel, tdict = modular_dict["sampler"].model.dump_all(metaargs=metaargs);
        if(not rot):
            proto=modular_dict["prototyper"](normproto);
        else:
            proto=modular_dict["prototyper"](normproto,rot);
        tdict[0]="";
        return  {"proto":proto,"plabel":plabel,"tdict":tdict};



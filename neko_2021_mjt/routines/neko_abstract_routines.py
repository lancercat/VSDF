import torch.nn.parallel
import torch



class neko_abstract_routine:

    def set_loggers(this,log_path,log_each,name):
        this.logger_dict={};
        pass
    def set_etc(this,args):
        pass;

    def __init__(this,args):
        mod_cvt_dicts, inp_cvt_dicts=\
        args["mod_cvt_dicts"],args["inp_cvt_dicts"]
        this.set_etc(args);
        this.name=args["name"];
        log_path, log_each=\
            args["log_path"],args["log_each"];
        this.log_each=log_each;

        # tells which module is which in modular_dict.
        # we may have two identical routines(DAN_char and DAN_word sharing only part of modules)
        this.mod_cvt_dict=mod_cvt_dicts;
        this.inp_cvt_dict=inp_cvt_dicts;
        this.set_loggers(log_path,log_each,args["name"])
        pass;
    def log_show(this):
        for k in this.logger_dict:
            this.logger_dict[k].show()

    def grab_nested(this,moduleterm,modular_dict):
        if (type(moduleterm) is list):
            return [this.grab_nested(n,modular_dict)for n in moduleterm]
        else:
            return modular_dict[moduleterm];
    def grab_modules(this,input_dict,modular_dict):
        mdict={};
        idict={}
        for k in this.mod_cvt_dict:
            mdict[k] = this.grab_nested(this.mod_cvt_dict[k],modular_dict);
        for k in this.inp_cvt_dict:
            idict[k] = input_dict[this.inp_cvt_dict[k]];
        return idict,mdict;

    def fp_impl(this,input_dict,modular_dict,logger_dict,nEpoch,batch_idx):
        return torch.tensor(0);

    def bp_impl(this, loss):
        loss.backward();

    def fpbp_impl(this,input_dict,modular_dict,logger_dict,nEpoch,batch_idx):
        loss=this.fp_impl(input_dict,modular_dict,logger_dict,nEpoch,batch_idx);
        this.bp_impl(loss);
        pass;
    def fpbp_amp_impl(this,input_dict,modular_dict,logger_dict,nEpoch,batch_idx):
        with torch.cuda.amp.autocast():
            loss=this.fp_impl(input_dict,modular_dict,logger_dict,nEpoch,batch_idx);
        this.bp_impl(loss);
        pass;

    def fp(this, input_dict, modular_dict, nEpoch, batch_idx):
        idict, mdict = this.grab_modules(input_dict, modular_dict);
        loss = this.fp_impl(idict, mdict, this.logger_dict, nEpoch, batch_idx);
        if (batch_idx % this.log_each == 0):
            this.log_show();
        return loss;

    def fpbp(this,input_dict,modular_dict,nEpoch,batch_idx):
        idict,mdict=this.grab_modules(input_dict,modular_dict);
        if("debug_path" in input_dict):
            idict["debug_path"]=input_dict["debug_path"];
        if("vdbg" in input_dict):
            idict["vdbg"] = input_dict["vdbg"];

        ret=this.fpbp_impl(idict,mdict,this.logger_dict,nEpoch,batch_idx);
        if (batch_idx % this.log_each == 0):
            this.log_show();
        return ret;
    def fpbp_amp(this,input_dict,modular_dict,nEpoch,batch_idx):
        idict,mdict=this.grab_modules(input_dict,modular_dict);
        ret=this.fpbp_amp_impl(idict,mdict,this.logger_dict,nEpoch,batch_idx);
        if (batch_idx % this.log_each == 0):
            this.log_show();
        return ret;
    def __call__(this, input_dict, modular_dict, nEpoch, batch_idx):
        this.fpbp(input_dict, modular_dict, nEpoch, batch_idx);
        return None;

# you may or may not sharing configs with training.
class neko_abstract_eval_routine:
    def clear_loggers(this):
        for l in this.logger_dict:
            this.logger_dict[l].clear();

    def set_loggers(this,log_path,name,args):
        this.logger_dict={};

    def set_etc(this, args):
        pass;

    def show_log(this):
        for lk in this.logger_dict:
            this.logger_dict[lk].show()
    def ret_log(this):
        ret={}
        for lk in this.logger_dict:
            ret[lk]=this.logger_dict[lk].show();
            this.logger_dict[lk].show()
    def __init__(this,args):
        this.set_etc(args);

        mod_cvt_dicts, inp_cvt_dicts=\
        args["mod_cvt_dicts"],args["inp_cvt_dicts"]
        log_path=args["log_path"];
        # tells which module is which in modular_dict.
        # we may have two identical routines(DAN_char and DAN_word sharing only part of modules)
        this.mod_cvt_dict=mod_cvt_dicts;
        this.inp_cvt_dict=inp_cvt_dicts;
        this.set_loggers(log_path,args["name"],args)
        pass;
    def interpret_mods(this,modular_dict):
        mdict={};
        for k in this.mod_cvt_dict:
            if (type(this.mod_cvt_dict[k]) is list):
                mdict[k]=[];
                for n in  this.mod_cvt_dict[k]:
                    # a weird string to ensure missing model is intentional
                    mdict[k].append(modular_dict[n])
            else:
                if (this.mod_cvt_dict[k] == "NEPnoneNEP"):
                    mdict[k] = None;
                else:
                    mdict[k] = modular_dict[this.mod_cvt_dict[k]];
        return mdict
    def grab_modules(this,input_dict,modular_dict):
        idict={}
        mdict=this.interpret_mods(modular_dict);
        for k in this.inp_cvt_dict:
            idict[k] = input_dict[this.inp_cvt_dict[k]];
        return idict,mdict;

    def pretest_impl(this,modular_dict,metaargs,**kwargs):
        rot = kwargs["rot"];
        normproto, plabel, tdict = modular_dict["sampler"].model.dump_all(metaargs=metaargs);
        if("[s]" in tdict):
            tdict[tdict["[s]"]]=0;
        if (not rot):
            proto = modular_dict["prototyper"](normproto);
        else:
            proto = modular_dict["prototyper"](normproto, rot);
        return {"proto":proto,"plabel":plabel,"tdict":tdict};



    def pretest(this,modular_dict,metaargs,override=None,**kwargs):
        mdict = this.interpret_mods( modular_dict);
        if (override is not None):
            for i in override:
                mdict[i] = modular_dict[i]
        return this.pretest_impl(mdict,metaargs, **kwargs);


    def test_impl(this,input_dict,modular_dict,logger_dict):
        pass;

    # return logits
    def vis_logits_impl(this,img,data_dict,modular_dict,at_time):
        pass;

    def vis_logit(this,img,input_dict,modular_dict,at_time,override=None,vdbg=None):
        idict, mdict = this.grab_modules(input_dict, modular_dict);
        if (not (vdbg is None)):
            idict["vdbg"] = vdbg;
        if (override is not None):
            for i in override:
                mdict[i] = modular_dict[i]
        return this.vis_logits_impl(img,idict, mdict,at_time);

    def test(this,input_dict,modular_dict,override=None,vdbg=None):
        idict,mdict=this.grab_modules(input_dict,modular_dict);
        if( not(vdbg is None)):
            idict["vdbg"]=vdbg;
        if(override is not None):
            for i in override:
                mdict[i]=modular_dict[i]
        return this.test_impl(idict,mdict,this.logger_dict);


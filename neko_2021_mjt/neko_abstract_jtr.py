import time

import torch
from torch import nn
import os
from neko_2020nocr.dan.common.common import display_cfgs,load_dataset,Zero_Grad,Train_or_Eval,generate_optimizer,Updata_Parameters,Updata_Parametersd,flatten_label
from neko_sdk.thirdparty.mmdetapply import multi_apply
import datetime
from torch.nn.parallel import parallel_apply

class neko_modular:
    def __init__(this,path,name,module,save_each=20000):
        this.path=path;
        this.model=module;
        this.name=name;
        this.save_each=save_each;
    def get_torch_module_dict(this):
        if(isinstance(this.model,nn.Module)):
            return this.model;
        else:
            return None;
    def train(this,training=True):
        this.model.train(training);
    def eval(this):
        this.model.eval();
    def normgrad(this):
        if this.save_each>0:
            nn.utils.clip_grad_norm_(this.model.parameters(), 20, 2)

    def cuda(this):
        this.model.cuda();

    def zero_grad(this):
        if this.save_each > 0:
            for param in this.model.parameters():
                param.grad = None
    def load(this,itrkey):
        p = this.path + itrkey + ".pth";
        try:
            this.model.load_state_dict(torch.load(p).state_dict())
        except:
            print(this.name, "cannot load", "itr",p,", starting fresh")

    def save(this,nEpoch):
        if(this.save_each>0 ):
            torch.save(this.model, this.path+'_E{}.pth'.format(nEpoch));
            torch.save(this.model, this.path + 'latest.pth');

    def save_if_needed(this,nEpoch,batch_idx):
        if(this.save_each>0 and batch_idx%this.save_each==0):
            print("Saving", this.path + '_E{}_I{}.pth'.format(nEpoch, batch_idx))
            torch.save(this.model,this.path+'_E{}_I{}.pth'.format(nEpoch,batch_idx));
            torch.save(this.model, this.path + 'latest.pth');

    def __call__(this, *args, **kwargs):
        return this.model(*args,**kwargs);


class neko_bogo_modular:
    def __init__(this,forwardable):
        # you should never haver a
        this.model=forwardable;
        this.save_each=-9
    def get_torch_module_dict(this):
        try:
            return this.model.get_torch_module_dict();
        except:
            return None;

        pass;
    def train(this,training=True):
        pass;

    def eval(this):
        pass;
    def normgrad(this):
        pass;

    def zero_grad(this):
        pass;

    def load(this,itrkey):
        pass;
    def cuda(this):
        pass;
    def save(this,nEpoch):
        pass;
    def save_if_needed(this,nEpoch,batch_idx):
        pass;

    def __call__(this, *args, **kwargs):
        return this.model(*args,**kwargs);
class neko_module_set:
    # if ("provides_bogo_modules" in cfg):
    #     this.bogo_modular_dict[name] = {}
    #     for k in cfg["provides_bogo_modules"]:
    #         this.bogo_modular_dict[name][k] = mod.bogo_modules[k];
    def arm_modules(this, root, modcfgs,itrkey):
        this.optimizers = [];
        this.optnames=[];
        this.optimizer_schedulers = [];
        this.modular_dict = {};
        this.bogo_modular_list=[]
        for name in modcfgs:
            cfg = modcfgs[name];
            modp=os.path.join(root, name);
            if("bogo_mod" in cfg):
                this.bogo_modular_list.append(name);
            else:
                mod, opt, opts = cfg["modular"](cfg["args"], modp, modp);
                this.modular_dict[name] = neko_modular(modp,name, mod, cfg["save_each"]);
                this.modular_dict[name].load(itrkey)
                if (opt is not None):
                    this.optimizers.append(opt);
                    this.optnames.append(name)
                    this.optimizer_schedulers.append(opts);
        # make sure we have collected real modules.
        for name in this.bogo_modular_list:
            cfg = modcfgs[name];
            # bogo modules are re-combination of parts of existing modules.
            mod= cfg["bogo_mod"](cfg["args"],this.modular_dict);
            this.modular_dict[name] = neko_bogo_modular(mod);

    def eval_mode(this):
        for modk in this.modular_dict:
            this.modular_dict[modk].eval();
        # this.modular_dict["sr_mjst_feature_extractor_cco"].train()
        # this.modular_dict["sr_mjst_LSCT_recon_loss"].train()
        # this.modular_dict["sr_mjst_CAM"].train()


    def train_mode(this):
        for modk in this.modular_dict:
            this.modular_dict[modk].train();
def update(opt):
    try:
        opt.step();
    except:
        print("Oops",opt);
        exit(9)
    return [];
def normgrad(mod):
    mod.normgrad();
    return [];
class neko_abstract_modular_joint_training(neko_module_set):

    def set_routines(this,routine_cfgs):
        this.routines=[];
        this.routine_names=[]
        for rcfg in routine_cfgs:
            this.routine_names.append(rcfg);
            this.routines.append(routine_cfgs[rcfg]["routine"](routine_cfgs[rcfg]))

    def set_val_tasks(this,val_cfgs):
        this.val_tasks = [];
        for vk in val_cfgs:
            this.val_tasks.append(val_cfgs[vk]["type"](None,None,this.modular_dict,val_cfgs[vk],1000))
    def set_dataloader(this,datacfg,vitr):
        this.joint_dataloader=datacfg["loadertype"](datacfg,vitr);
    def setup(this,cfgs):
        root, this.val_each, this.vitr, this.vepoch = \
            cfgs["root"], cfgs["val_each"], cfgs["vitr"], cfgs["vepoch"];
        # set to "latest" for resuming, whatever does not make sense to start fresh.
        this.set_dataloader(cfgs["dataloader_cfg"], vitr=cfgs["vitr"]);
        this.arm_modules(root, cfgs["modules"], cfgs["iterkey"]);
        this.set_routines(cfgs["routine_cfgs"]);
        this.set_val_tasks(cfgs["tasks"]);

    def __init__(this,
                 cfgs):
        this.setup(cfgs)
        pass;

        # ---------------------------------
    def val(this,nEpoch,batch_idx,vdbg=None):
        this.eval_mode()
        # torch.cuda.empty_cache();
        for vt in this.val_tasks:
            print(nEpoch,batch_idx);
            torch.cuda.empty_cache();
            with torch.no_grad():
                vt.test(vdbg=vdbg);
        torch.cuda.empty_cache();
        this.train_mode();
    def launch(this,rot,sample_batched,nEpoch,batch_idx):
        rot.fpbp(sample_batched, this.modular_dict, nEpoch, batch_idx)
        return []
    def tr_iter_amp(this,nEpoch,batch_idx,sample_batched):
        # torch.autograd.set_detect_anomaly(True);
        # data prepare
        zg_start=time.time();
        for modk in this.modular_dict:
            this.modular_dict[modk].zero_grad();

        routine_start=time.time();
        # multi_apply(this.launch,this.routines, sample_batched=sample_batched, nEpoch=nEpoch,
        #             batch_idx=batch_idx)
        #
        for routine in this.routines:
            routine.fpbp_amp(sample_batched,this.modular_dict,nEpoch,batch_idx)
        # reqnorm=[]
        # for modk in this.modular_dict:
        #     if(this.modular_dict[modk].save_each>0):
        #         reqnorm.append(this.modular_dict[modk]);
        # multi_apply(normgrad,reqnorm)

        for modk in this.modular_dict:
            if(this.modular_dict[modk].save_each>0):
                ng_start_ = time.time();
                this.modular_dict[modk].normgrad();
                # print(modk,time.time()-ng_start_);
        pu_start=time.time();
        try:
            Updata_Parameters(this.optimizers, frozen=[]);
        except:
            print("Oops");
            exit(9);
        all_done=time.time();

        if(batch_idx%100==9):
            print("[Timings]: zg:",routine_start-zg_start, "routines:", pu_start-routine_start,"pu:",all_done-pu_start);

        for modk in this.modular_dict:
            this.modular_dict[modk].save_if_needed(nEpoch,batch_idx);

    def tr_iter(this,nEpoch,batch_idx,sample_batched):
        # torch.autograd.set_detect_anomaly(True);
        # data prepare
        zg_start=time.time();
        for modk in this.modular_dict:
            this.modular_dict[modk].zero_grad();

        routine_start=time.time();
        # multi_apply(this.launch,this.routines, sample_batched=sample_batched, nEpoch=nEpoch,
        #             batch_idx=batch_idx)
        #
        for routine in this.routines:
            routine.fpbp(sample_batched,this.modular_dict,nEpoch,batch_idx)
        # reqnorm=[]
        # for modk in this.modular_dict:
        #     if(this.modular_dict[modk].save_each>0):
        #         reqnorm.append(this.modular_dict[modk]);
        # multi_apply(normgrad,reqnorm)

        for modk in this.modular_dict:
            if(this.modular_dict[modk].save_each>0):
                ng_start_ = time.time();
                this.modular_dict[modk].normgrad();
                # print(modk,time.time()-ng_start_);
        pu_start=time.time();
        try:
            Updata_Parameters(this.optimizers, frozen=[]);
        except:
            print("Oops");
            exit(9);
        all_done=time.time();

        if(batch_idx%100==9):
            print("[Timings]: zg:",routine_start-zg_start, "routines:", pu_start-routine_start,"pu:",all_done-pu_start);

        for modk in this.modular_dict:
            this.modular_dict[modk].save_if_needed(nEpoch,batch_idx);


    def train(this,dbgpath,vdbg=None,flag=None):
        torch.backends.cudnn.benchmark=True;

        for modk in this.modular_dict:
            this.modular_dict[modk].cuda();
        for modk in this.modular_dict:
            this.modular_dict[modk].train();
        for nEpoch in range(0, this.vepoch):
            for batch_idx in range(this.vitr):
                if(flag is None or flag==False):
                    flag = (batch_idx > 0) or (dbgpath is not None);
                if (flag and batch_idx % this.val_each == 0):
                    this.val(nEpoch, batch_idx,vdbg=vdbg);
                data_start=time.time();
                sample_batched=this.joint_dataloader.next();
                data_time=time.time()-data_start;
                itr_start=time.time();
                if(dbgpath is not None):
                    sample_batched["debug_path"]=dbgpath;
                if(vdbg is not None):
                    sample_batched["vdbg"]=vdbg;

                # for d in sample_batched:
                #     if(type(sample_batched[d])==torch.tensor):
                #         sample_batched[d]=sample_batched[d].cuda()
                this.tr_iter(nEpoch,batch_idx,sample_batched)
                itr_time = time.time()-itr_start;

                # print(torch.backends.cudnn.benchmark);
                if(batch_idx%100==9):
                    print("datatime",data_time,"itrtime",itr_time,"all",time.time()-data_start);
            Updata_Parameters(this.optimizer_schedulers, frozen=[])
            this.val(nEpoch, "Final");
            # torch.backends.cudnn.benchmark = False;
            for modk in this.modular_dict:
                this.modular_dict[modk].save(nEpoch);
class neko_abstract_modular_joint_eval(neko_module_set):

    def set_val_tasks(this,val_cfgs,mitr):
        this.val_tasks = [];
        for vk in val_cfgs:
            this.val_tasks.append(val_cfgs[vk]["type"](None,None,this.modular_dict,val_cfgs[vk],mitr))
    def test_img(this,id,image_path,globalcache,h=32,w=100):
        return this.val_tasks[id].test_image(image_path,globalcache)
    def pretest(this,id):
        this.eval_mode();
        with torch.no_grad():
            tester=this.val_tasks[id].testready();
        return tester;

    def __init__(this,
                 cfgs,mitr):
        root= \
        cfgs["root"];
        # set to "latest" for resuming, whatever does not make sense to start fresh.
        this.arm_modules(root,cfgs["modules"],cfgs["iterkey"]);
        for mk in this.modular_dict:
            this.modular_dict[mk].model.cuda();
        if("export_path" in cfgs and cfgs["export_path"] is not None):
            for k in cfgs["tasks"]:
                cfgs["tasks"][k]["export_path"]=cfgs["export_path"];
        this.set_val_tasks(cfgs["tasks"],mitr);
        pass;

        # ---------------------------------
    def val(this,nEpoch,batch_idx,rot=0):
        this.eval_mode()
        for vt in this.val_tasks:
            print(nEpoch,batch_idx);
            torch.cuda.empty_cache();
            with torch.no_grad():
                vt.test(rot);
        this.train_mode()

    def vis(this, nEpoch, batch_idx, rot=0):
        this.eval_mode()
        for vt in this.val_tasks:
            print(nEpoch, batch_idx);
            torch.cuda.empty_cache();
            vt.visualize(rot);
        this.train_mode()

        # ---------------------------------
    def valt(this,nEpoch,batch_idx):
        this.train_mode()
        for vt in this.val_tasks:
            print(nEpoch,batch_idx);
            with torch.no_grad():
                torch.cuda.empty_cache()
                vt.test();
        this.train_mode()
# some routines may change shared module state.
class neko_modular_joint_training_semipara(neko_abstract_modular_joint_training):
    def tr_iter(this,nEpoch,batch_idx,sample_batched):
        # torch.autograd.set_detect_anomaly(True);
        # data prepare
        # torch.backends.cudnn.benchmark=True;

        zg_start=time.time();

        for modk in this.modular_dict:
            this.modular_dict[modk].zero_grad();
        routine_start=time.time();

        # multi_apply(this.launch,this.routines, sample_batched=sample_batched, nEpoch=nEpoch,
        #             batch_idx=batch_idx)

        # i=0;
        for routine in this.routines:
            # rs = time.time();
            routine.fpbp(sample_batched,this.modular_dict,nEpoch,batch_idx)
        #
        #     # print(this.routine_names[i],time.time()-rs);
        #     # i += 1;
        ng_start=time.time();

        for modk in this.modular_dict:
            if(this.modular_dict[modk].save_each>0):
                this.modular_dict[modk].normgrad();
        pu_start=time.time();
        multi_apply(update,this.optimizers);
        # try:
        #     Updata_Parametersd(this.optimizers,this.optnames, frozen=[]);
        # except:
        #     print("Oops");
        #     exit(9);
        all_done=time.time();
        if (batch_idx % 100 == 9):
            print("[Timings]: zg:", routine_start - zg_start, "routines:", pu_start - routine_start, "pu:", all_done - pu_start);

        for modk in this.modular_dict:
            this.modular_dict[modk].save_if_needed(nEpoch,batch_idx);
class neko_modular_joint_training_para(neko_abstract_modular_joint_training):
    def tr_iter(this,nEpoch,batch_idx,sample_batched):
        # torch.autograd.set_detect_anomaly(True);
        # data prepare
        # torch.backends.cudnn.benchmark=True;

        zg_start=time.time();

        for modk in this.modular_dict:
            this.modular_dict[modk].zero_grad();
        routine_start=time.time();

        multi_apply(this.launch,this.routines, sample_batched=sample_batched, nEpoch=nEpoch,
                    batch_idx=batch_idx)

        # i=0;
        # for routine in this.routines:
        #     # rs = time.time();
        #     routine.fpbp(sample_batched,this.modular_dict,nEpoch,batch_idx)
        #
        #     # print(this.routine_names[i],time.time()-rs);
        #     # i += 1;
        ng_start=time.time();

        # for modk in this.modular_dict:
        #     if(this.modular_dict[modk].save_each>0):
        #         this.modular_dict[modk].normgrad();
        pu_start=time.time();
        multi_apply(update,this.optimizers);
        # try:
        #     Updata_Parameters(this.optimizers, frozen=[]);
        # except:
        #     print("Oops");
        #     exit(9);
        all_done=time.time();
        if (batch_idx % 100 == 9):
            print("[Timings]: zg:", routine_start - zg_start, "routines:", pu_start - routine_start, "pu:", all_done - pu_start);

        for modk in this.modular_dict:
            this.modular_dict[modk].save_if_needed(nEpoch,batch_idx);

class neko_modular_joint_training_para2(neko_abstract_modular_joint_training):
    def launch(this,rot,sample_batched,nEpoch,batch_idx):
        l=rot.fp(sample_batched, this.modular_dict, nEpoch, batch_idx)
        return [l]
    def tr_iter(this,nEpoch,batch_idx,sample_batched):
        # torch.autograd.set_detect_anomaly(True);
        # data prepare
        # torch.backends.cudnn.benchmark=True;

        zg_start=time.time();

        for modk in this.modular_dict:
            this.modular_dict[modk].zero_grad();
        routine_start=time.time();

        losses=multi_apply(this.launch,this.routines, sample_batched=sample_batched, nEpoch=nEpoch,
                    batch_idx=batch_idx)
        loss=torch.stack([loss[0] for loss in losses]).sum();

        #
        # loss=0;
        # for routine in this.routines:
        #     # rs = time.time();
        #     loss+=routine.fp(sample_batched,this.modular_dict,nEpoch,batch_idx)
        #     # print(this.routine_names[i],time.time()-rs);
        #     # i += 1;
        #
        loss.backward();
        ng_start=time.time();

        for modk in this.modular_dict:
            if(this.modular_dict[modk].save_each>0):
                this.modular_dict[modk].normgrad();
        pu_start=time.time();
        # multi_apply(update,this.optimizers);
        try:
            Updata_Parameters(this.optimizers, frozen=[]);
        except:
            print("Oops");
        #     exit(9);
        all_done=time.time();
        if (batch_idx % 100 == 9):
            print("[Timings]: zg:", routine_start - zg_start, "routines:", pu_start - routine_start, "pu:", all_done - pu_start);

        for modk in this.modular_dict:
            this.modular_dict[modk].save_if_needed(nEpoch,batch_idx);
class neko_modular_joint_training_para3(neko_abstract_modular_joint_training):
    def launch(this,rot,sample_batched,nEpoch,batch_idx):
        l=rot.fp(sample_batched, this.modular_dict, nEpoch, batch_idx)
        return [l]
    def tr_iter(this,nEpoch,batch_idx,sample_batched):
        # torch.autograd.set_detect_anomaly(True);
        # data prepare
        # torch.backends.cudnn.benchmark=True;

        zg_start=time.time();

        for modk in this.modular_dict:
            this.modular_dict[modk].zero_grad();
        routine_start=time.time();
        inp=[[sample_batched,this.modular_dict,nEpoch,batch_idx] for _ in this.routines]
        dev=["cuda" for _ in this.routines];
        parallel_apply(this.routines,inp,devices=dev)

        #
        # loss=0;
        # for routine in this.routines:
        #     # rs = time.time();
        #     loss+=routine.fp(sample_batched,this.modular_dict,nEpoch,batch_idx)
        #     # print(this.routine_names[i],time.time()-rs);
        #     # i += 1;
        #
        # loss.backward();
        ng_start=time.time();

        for modk in this.modular_dict:
            if(this.modular_dict[modk].save_each>0):
                this.modular_dict[modk].normgrad();
        pu_start=time.time();
        # multi_apply(update,this.optimizers);
        try:
            Updata_Parameters(this.optimizers, frozen=[]);
        except:
            print("Oops");
        #     exit(9);
        all_done=time.time();
        if (batch_idx % 100 == 9):
            print("[Timings]: zg:", routine_start - zg_start, "routines:", pu_start - routine_start, "pu:", all_done - pu_start);

        for modk in this.modular_dict:
            this.modular_dict[modk].save_if_needed(nEpoch,batch_idx);

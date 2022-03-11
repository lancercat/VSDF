
from neko_2021_mjt.configs.common_subs.arm_postfe import arm_rest_commonwops
from neko_2021_mjt.configs.modules.config_cam import config_cam
from  neko_2021_mjt.configs.modules.config_fe_db import \
    config_fe_r45_binorm_orig,config_fe_r45_binorm_ptpt
from neko_2021_mjt.configs.modules.config_semantic_branches import \
    config_sampled_semantic_branch
from neko_2021_mjt.configs.modules.config_ocr_sampler_gl import \
    config_gocr_sampler
from neko_2021_mjt.configs.bogo_modules.config_res_binorm import \
    config_bogo_resbinorm;
from neko_2021_mjt.configs.common_subs.arm_post_fe_shared_prototyper import\
    arm_shared_prototyper,arm_shared_prototyper_np
from neko_2021_mjt.configs.modules.config_sa import config_sa_mk2,config_sa_mk3
from neko_2021_mjt.configs.modules.config_ctx_module import  \
    config_ctxmodule
from neko_2021_mjt.configs.modules.config_cam_stop import \
    config_cam_stop
from neko_2021_mjt.configs.modules.config_cls_emb_loss import\
    config_cls_emb_loss2,config_cls_lossohem
from neko_2021_mjt.eval_tasks.dan_eval_tasks import neko_odan_eval_tasks_mk8
# please note it exists, but not used, it can be buggy or not if you use it(
# depends on whether we fixed it for good), we don't really know for now.
# That's why this is not included in the paper.
from neko_2021_mjt.configs.modules.config_vpystat import config_vpystat;


def arm_trinorm_mk8_common_noa(srcdst,prefix,capacity,feat_ch,tr_meta_path,expf=1,views=["synthw","glyph"],ccnt=3900,camch=64):
    srcdst[prefix + "feature_extractor_container"] = config_fe_r45_binorm_orig(3, feat_ch, cnt=len(views),expf=expf);
    for i in range(len(views)):
        srcdst[prefix + "feature_extractor" + views[i]] = config_bogo_resbinorm(prefix + "feature_extractor_container",
                                                                                "res" + str(i + 1))
    srcdst[prefix + "sampler"] = config_gocr_sampler(tr_meta_path, capacity);
    srcdst[prefix + "semantic_branch"] = config_sampled_semantic_branch(ccnt, feat_ch);
    srcdst[prefix+ "vpystat"]=config_vpystat()
    return srcdst;
def arm_trinorm_mk8_ptpt_common_noa(srcdst,prefix,capacity,feat_ch,tr_meta_path,expf=1,views=["synthw","glyph"],ccnt=3900,camch=64):
    srcdst[prefix + "feature_extractor_container"] = config_fe_r45_binorm_ptpt(3, feat_ch, cnt=len(views),expf=expf);
    for i in range(len(views)):
        srcdst[prefix + "feature_extractor" + views[i]] = config_bogo_resbinorm(prefix + "feature_extractor_container",
                                                                                "res" + str(i + 1))
    srcdst[prefix + "sampler"] = config_gocr_sampler(tr_meta_path, capacity);
    srcdst[prefix + "semantic_branch"] = config_sampled_semantic_branch(ccnt, feat_ch);
    srcdst[prefix+ "vpystat"]=config_vpystat()
    return srcdst;
def arm_trinorm_mk8_common_nota(srcdst,prefix,capacity,feat_ch,tr_meta_path,expf=1,views=["synthw","glyph"],ccnt=3900,camch=64):
    srcdst=arm_trinorm_mk8_common_noa(srcdst,prefix,capacity,feat_ch,tr_meta_path,expf=expf,views=views,ccnt=ccnt,camch=camch);
    srcdst[prefix + "GA"] = config_sa_mk2(feat_ch=128);
    return srcdst;
def arm_trinorm_mk8h_common_nota(srcdst,prefix,capacity,feat_ch,tr_meta_path,expf=1,views=["synthw","glyph"],ccnt=3900,camch=64):
    srcdst=arm_trinorm_mk8_common_noa(srcdst,prefix,capacity,feat_ch,tr_meta_path,expf=expf,views=views,ccnt=ccnt,camch=camch);
    srcdst[prefix + "GA"] = config_sa_mk3(feat_ch=int(32*expf));
    return srcdst;

def arm_trinorm_mk8h_common_ptpt_nota(srcdst,prefix,capacity,feat_ch,tr_meta_path,expf=1,views=["synthw","glyph"],ccnt=3900,camch=64):
    srcdst=arm_trinorm_mk8_ptpt_common_noa(srcdst,prefix,capacity,feat_ch,tr_meta_path,expf=expf,views=views,ccnt=ccnt,camch=camch);
    srcdst[prefix + "GA"] = config_sa_mk3(feat_ch=int(64*expf));
    return srcdst;



def arm_trinorm_mk8_common(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1,views=["synthw","glyph"],ccnt=3900,camch=64):
    srcdst=arm_trinorm_mk8_common_nota(srcdst,prefix,capacity,feat_ch,tr_meta_path,expf=expf,views=views,ccnt=ccnt,camch=camch)
    srcdst[prefix + "TA"] = config_cam_stop(maxT, feat_ch=feat_ch, scales=[
        [int(32), 16, 64],
        [int(128), 8, 32],
        [int(feat_ch), 8, 32]
    ],cam_ch=camch);
    return srcdst;
def arm_trinorm_mk8h_common(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1,views=["synthw","glyph"],ccnt=3900,camch=64):
    srcdst=arm_trinorm_mk8h_common_nota(srcdst,prefix,capacity,feat_ch,tr_meta_path,expf=expf,views=views,ccnt=ccnt,camch=camch)
    if(maxT==1):
        maxT=2;
    srcdst[prefix + "TA"] = config_cam_stop(maxT, feat_ch=feat_ch, scales=[
        [int(32*expf), 16, 64],
        [int(128*expf), 8, 32],
        [int(feat_ch), 8, 32]
    ],cam_ch=camch);
    return srcdst;
def arm_trinorm_mk8h_ptpt_common(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1,views=["synthw","glyph"],ccnt=3900,camch=64):
    srcdst=arm_trinorm_mk8h_common_ptpt_nota(srcdst,prefix,capacity,feat_ch,tr_meta_path,expf=expf,views=views,ccnt=ccnt,camch=camch)
    if(maxT==1):
        maxT=2;
    srcdst[prefix + "TA"] = config_cam_stop(maxT, feat_ch=feat_ch, scales=[
        [int(64*expf), 16, 64],
        [int(256*expf), 8, 32],
        [int(feat_ch), 8, 32]
    ],cam_ch=camch);
    return srcdst;

def arm_mk8_proto_sga(srcdst,prefix,capacity,feat_ch):
    srcdst = arm_shared_prototyper(
        srcdst, prefix, capacity, feat_ch,
        prefix + "feature_extractor" + "glyph",
        prefix + "GA",
        use_sp=False,
        nameoverride=prefix + "prototyper" + "glyph",
    );
    return srcdst;
# fixed the feat grad cut feature/bug in the prototyper.
def arm_mk8_proto_sganp(srcdst,prefix,capacity,feat_ch):
    srcdst = arm_shared_prototyper_np(
        srcdst, prefix, capacity, feat_ch,
        prefix + "feature_extractor" + "glyph",
        prefix + "GA",
        use_sp=False,
        nameoverride=prefix + "prototyper" + "glyph",
    );
    return srcdst;


def arm_mk8_ctx(srcdst,prefix,feat_ch,nhead=8,nlay=4):
    srcdst[prefix+"ctxloss"]=  config_cls_emb_loss2();
    srcdst[prefix+"ctxmodule"]=config_ctxmodule(feat_ch,nhead,nlay);
    return srcdst;


def arm_trinorm_mk8hnp_module_set_dan_r45ptpt(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1,views=["synthw","glyph"],ccnt=3900,camch=64,sheads=8,slays=4,wemb=0.3):
    srcdst=arm_trinorm_mk8h_ptpt_common(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=expf,views=views,ccnt=ccnt,camch=camch);
    srcdst=arm_mk8_proto_sganp(srcdst,prefix,capacity,feat_ch);
    srcdst = arm_rest_commonwops(srcdst,prefix,wemb=wemb);
    srcdst=arm_mk8_ctx(srcdst,prefix,feat_ch,sheads,slays);
    return srcdst;

def arm_trinorm_mk8hnp_module_set_dan_r45(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1,views=["synthw","glyph"],ccnt=3900,camch=64,sheads=8,slays=4,wemb=0.3):
    srcdst=arm_trinorm_mk8h_common(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=expf,views=views,ccnt=ccnt,camch=camch);
    srcdst=arm_mk8_proto_sganp(srcdst,prefix,capacity,feat_ch);
    srcdst = arm_rest_commonwops(srcdst,prefix,wemb=wemb);
    srcdst=arm_mk8_ctx(srcdst,prefix,feat_ch,sheads,slays);
    return srcdst;



def arm_GTA_mk8_routine(srcdst, prefix, routine_type,maxT, log_path, log_each,dsprefix,view_name,proto_viewname):
    srcdst[prefix+"mjst"]= routine_type(
        prototyper_name=prefix+"prototyper"+proto_viewname,
        sampler_name=prefix+"sampler",
        vpystat_name=prefix+"vpystat",
        semantic_branch_name=prefix+"semantic_branch",
        feature_extractor_name=prefix+"feature_extractor"+view_name,
        GA_name=prefix+"GA",
        TA_name=prefix+"TA",
        seq_name=prefix+"DTD",
        pred_name=[prefix+"pred"],
        ctxmodule_name=prefix+"ctxmodule",
        ctxloss_name=prefix + "ctxloss",
        loss_name=[prefix+"loss_cls_emb"],
        image_name=dsprefix+"image",
        label_name=dsprefix+"label",
        mask_name=dsprefix+"bmask",
        log_path=log_path,
        log_each=log_each,
        name=prefix+"mjst",
        maxT=maxT,
    );
    srcdst[prefix+"mjst"]["stream"]=prefix;
    return srcdst;

def arm_base_mk8_routine(srcdst, prefix, routine_type,maxT, log_path, log_each,dsprefix,view_name,proto_viewname):
    srcdst[prefix+"mjst"]= routine_type(
        prototyper_name=prefix+"prototyper"+proto_viewname,
        sampler_name=prefix+"sampler",
        semantic_branch_name=prefix+"semantic_branch",
        feature_extractor_name=prefix+"feature_extractor"+view_name,
        TA_name=prefix+"TA",
        seq_name=prefix+"DTD",
        pred_name=[prefix+"pred"],
        ctxmodule_name=prefix+"ctxmodule",
        loss_name=[prefix+"loss_cls_emb"],
        ctxloss_name=prefix+"ctxloss",
        image_name=dsprefix+"image",
        mask_name=dsprefix+"bmask",
        label_name=dsprefix+"label",
        log_path=log_path,
        log_each=log_each,
        name=prefix+"mjst",
        maxT=maxT,
    );
    srcdst[prefix+"mjst"]["stream"]=prefix;
    return srcdst;


def arm_base_mk8_eval_routine(srcdst,tname,prefix,routine_type,log_path,maxT,view_name,proto_viewname,force_skip_ctx,measure_rej):
    if(force_skip_ctx):
        # make a weird string to make we only skip a module intentionally.
        # This also helps us locate corresponding code faster
        ctx_name="NEPnoneNEP"
    else:
        ctx_name=prefix + "ctxmodule"
    return routine_type(
        prototyper_name=prefix+"prototyper"+proto_viewname,
        sampler_name=prefix+"sampler",
        semantic_branch_name=prefix+"semantic_branch",
        feature_extractor_name=prefix+"feature_extractor"+view_name,
        TA_name=prefix+"TA",
        seq_name=prefix+"DTD",
        pred_name=[prefix+"pred"],
        ctxmodule_name=ctx_name,
        loss_name=[prefix + "loss_cls_emb"],
        image_name="image",
        mask_name=  "bmask",
        label_name="label",
        log_path=log_path,
        name=prefix + tname,
        maxT=maxT,
        measure_rej=measure_rej,
    );


def arm_base_mk8_task_default(srcdst,prefix,routine_type,maxT,te_meta_path,datasets,log_path,name="close_set_benchmarks",view_name="synthw",proto_viewname="glyph",force_skip_ctx=False,measure_rej=False):
    te_routine={};
    te_routine=arm_base_mk8_eval_routine(te_routine,"close_set_benchmark",prefix,routine_type,log_path,maxT,view_name=view_name,proto_viewname=proto_viewname,force_skip_ctx=force_skip_ctx,measure_rej=measure_rej)
    srcdst[prefix+name]={
        "type": neko_odan_eval_tasks_mk8,
        "protoname": prefix+"prototyper"+"glyph",
        "temeta":
            {
                "meta_path": te_meta_path,
                "case_sensitive": False,
            },
        "datasets":datasets,
        "routine_cfgs": te_routine,
    }
    return srcdst

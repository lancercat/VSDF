from neko_2021_mjt.configs.modules.config_fe_cco import config_fe_cco_thicc
from  neko_2021_mjt.configs.modules.config_fe_std import config_fe_r45

from neko_2021_mjt.configs.common_subs.arm_postfe import arm_rest_common
from neko_2021_mjt.configs.modules.config_cam import config_cam



def arm_base_module_set_dan_r45(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1,wemb=0.3):
    srcdst[prefix+"feature_extractor_cco"]= config_fe_r45(3, feat_ch);
    srcdst[prefix+"CAM"]= config_cam(maxT, feat_ch=feat_ch,scales=[
                    [int(32), 16, 64],
                    [int(128), 8, 32],
                    [int(feat_ch), 8, 32]
                ]);
    srcdst = arm_rest_common(srcdst, prefix, maxT, capacity, feat_ch, tr_meta_path,wemb=wemb)
    return srcdst;

def arm_base_module_set_thicc(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path):
    srcdst[prefix+"feature_extractor_cco"]= config_fe_cco_thicc(3, feat_ch);
    srcdst[prefix+"CAM"]= config_cam(maxT, feat_ch=feat_ch, expf=1.5);
    srcdst = arm_rest_common(srcdst, prefix, maxT, capacity, feat_ch, tr_meta_path)
    return srcdst;




def arm_base_routine(srcdst, prefix, routine_type,maxT, log_path, log_each,dsprefix):
    srcdst[prefix+"mjst"]= routine_type(
        prototyper_name=prefix+"prototyper",
        sampler_name=prefix+"Latin_62_sampler",
        feature_extractor_name=prefix+"feature_extractor_cco",
        CAMname=prefix+"CAM",
        seq_name=prefix+"DTD",
        pred_name=[prefix+"pred"],
        loss_name=[prefix+"loss_cls_emb"],
        image_name=dsprefix+"image",
        label_name=dsprefix+"label",
        log_path=log_path,
        log_each=log_each,
        name=prefix+"mjst",
        maxT=maxT,
    );
    srcdst[prefix+"mjst"]["stream"]=prefix;
    return srcdst;



def arm_base_routine2(srcdst, prefix, routine_type,maxT, log_path, log_each,dsprefix):
    srcdst[prefix+"mjst"]= routine_type(
        prototyper_name=prefix+"prototyper",
        sampler_name=prefix+"Latin_62_sampler",
        feature_extractor_name=prefix+"feature_extractor_cco",
        CAMname=prefix+"TA",
        seq_name=prefix+"DTD",
        pred_name=[prefix+"pred"],
        loss_name=[prefix+"loss_cls_emb"],
        image_name=dsprefix+"image",
        label_name=dsprefix+"label",
        log_path=log_path,
        log_each=log_each,
        name=prefix+"mjst",
        maxT=maxT,
    );
    srcdst[prefix+"mjst"]["stream"]=prefix;
    return srcdst;

def arm_base_eval_routine(srcdst,tname,prefix,routine_type,log_path,maxT):
    return routine_type(
        prototyper_name=prefix + "prototyper",
        sampler_name=prefix + "Latin_62_sampler",
        feature_extractor_name=prefix + "feature_extractor_cco",
        CAMname=prefix + "CAM",
        seq_name=prefix + "DTD",
        pred_name=[prefix + "pred"],
        loss_name=[prefix + "loss_cls_emb"],
        image_name="image",
        label_name="label",
        log_path=log_path,
        name=prefix + tname,
        maxT=maxT,
    );
def arm_base_eval_routine2(srcdst,tname,prefix,routine_type,log_path,maxT,measure_rej=False):
    return routine_type(
        prototyper_name=prefix + "prototyper",
        sampler_name=prefix + "Latin_62_sampler",
        feature_extractor_name=prefix + "feature_extractor_cco",
        CAMname=prefix + "TA",
        seq_name=prefix + "DTD",
        pred_name=[prefix + "pred"],
        loss_name=[prefix + "loss_cls_emb"],
        image_name="image",
        label_name="label",
        log_path=log_path,
        name=prefix + tname,
        maxT=maxT,
        measure_rej=measure_rej,
    );
from neko_2021_mjt.eval_tasks.dan_eval_tasks import neko_odan_eval_tasks

def arm_base_task_default(srcdst,prefix,routine_type,maxT,te_meta_path,datasets,log_path,name="close_set_benchmarks"):
    te_routine={};
    te_routine=arm_base_eval_routine(te_routine,"close_set_benchmark",prefix,routine_type,log_path,maxT)
    srcdst[prefix+name]={
        "type": neko_odan_eval_tasks,
        "protoname": prefix+"prototyper",
        "temeta":
            {
                "meta_path": te_meta_path,
                "case_sensitive": False,
            },
        "datasets":datasets,
        "routine_cfgs": te_routine,
    }
    return srcdst
def arm_base_task_default2(srcdst,prefix,routine_type,maxT,te_meta_path,datasets,log_path,name="close_set_benchmarks",measure_rej=False):
    te_routine={};
    te_routine=arm_base_eval_routine2(te_routine,"close_set_benchmark",prefix,routine_type,log_path,maxT,measure_rej=measure_rej)
    srcdst[prefix+name]={
        "type": neko_odan_eval_tasks,
        "protoname": prefix+"prototyper",
        "temeta":
            {
                "meta_path": te_meta_path,
                "case_sensitive": False,
            },
        "datasets":datasets,
        "routine_cfgs": te_routine,
    }
    return srcdst
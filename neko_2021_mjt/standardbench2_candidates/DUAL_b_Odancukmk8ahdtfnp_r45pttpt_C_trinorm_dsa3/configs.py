from neko_2021_mjt.configs.loadouts.mk8.base_mk8_module_set import arm_base_mk8_routine,arm_base_mk8_task_default

from neko_2021_mjt.configs.loadouts.mk8.base_mk8_module_set import arm_trinorm_mk8hnp_module_set_dan_r45ptpt

from neko_2021_mjt.configs.routines.ocr_routines.mk8.osdanmk8_routine_cfg import osdanmk8adt_ocr_routine
from neko_2021_mjt.configs.routines.ocr_routines.mk8.osdanmk8_routine_cfg import  osdanmk8_eval_routine_cfg
from neko_2021_mjt.dss_presets.dual_no_lsct_32 import get_dss;


def model_mod_cfg(tr_meta_path_chs,tr_meta_path_mjst,maxT_mjst,maxT_chs):
    capacity=256;
    feat_ch=512;
    mods={};
    mods=arm_trinorm_mk8hnp_module_set_dan_r45ptpt(mods,"base_mjst_",maxT_mjst,capacity,feat_ch,tr_meta_path_mjst,ccnt=38,wemb=0,expf=1.5);
    # mods=arm_trinorm_mk8hnp_module_set_dan_r45(mods,"base_chs_",maxT_chs,capacity,feat_ch,tr_meta_path_chs,ccnt=3824,wemb=0);
    return mods;


def dan_single_model_train_cfg(save_root,dsroot,
                               log_path,log_each,itrk= "Top Nep",bsize=48,tvitr=200000):
    maxT_chs=30;
    maxT_mjst=25;

    tr_meta_path_chsjap, te_meta_path_chsjap, tr_meta_path_mjst,te_meta_path_mjst, \
    mjst_eval_ds, chs_eval_ds, train_joint_ds=get_dss(dsroot,maxT_mjst,maxT_chs,bsize);

    task_dict = {}
    task_dict = arm_base_mk8_task_default(task_dict, "base_mjst_", osdanmk8_eval_routine_cfg, maxT_mjst, te_meta_path_mjst,mjst_eval_ds , log_path,force_skip_ctx=False);
    # task_dict = arm_base_mk8_task_default(task_dict, "base_chs_", osdanmk8_eval_routine_cfg, maxT_chs, te_meta_path_chsjap, chs_eval_ds,
    #                                   log_path,force_skip_ctx=True);

    routines = {};
    routines = arm_base_mk8_routine(routines, "base_mjst_", osdanmk8adt_ocr_routine, maxT_mjst, log_path,
                                     log_each, "dan_mjst_", view_name="synthw", proto_viewname="glyph");
    # routines = arm_base_mk8_routine(routines, "base_chs_", osdanmk8adt_ocr_routine, maxT_chs, log_path,
    #                                  log_each, "dan_chs_", view_name="synthw", proto_viewname="glyph");

    return \
        {
            "root": save_root,
            "val_each": 10000,
            "vitr": 200000,
            "vepoch": 4,
            "iterkey": itrk,  # something makes no sense to start fresh
            "dataloader_cfg":train_joint_ds,
            # make sure the unseen characters are unseen.
            "modules": model_mod_cfg(tr_meta_path_chsjap,tr_meta_path_mjst, maxT_mjst,maxT_chs),
            "routine_cfgs": routines,
            "tasks": task_dict,
        }
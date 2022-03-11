from neko_2021_mjt.configs.loadouts.mk8.base_mk8_module_set import arm_base_mk8_routine,arm_base_mk8_task_default

from neko_2021_mjt.dss_presets.dual_chhwctw_32 import get_dss;
from neko_2021_mjt.configs.loadouts.mk8.base_mk8_module_set import arm_trinorm_mk8hnp_module_set_dan_r45
from neko_2021_mjt.configs.routines.ocr_routines.mk8.osdanmk8_routine_cfg import osdanmk8adt_ocr_routine
from neko_2021_mjt.configs.routines.ocr_routines.mk8.osdanmk8_routine_cfg import  osdanmk8_eval_routine_cfg

def model_mod_cfg(tr_meta_path_ctw,tr_meta_path_hwdb):
    capacity=256;
    feat_ch=512;
    mods={};

    # mods=arm_trinorm_mk8hnp_module_set_dan_r45(mods,"base_hwdb_",1,capacity,feat_ch,tr_meta_path_hwdb,wemb=0);
    mods=arm_trinorm_mk8hnp_module_set_dan_r45(mods,"base_ctw_",1,capacity,feat_ch,tr_meta_path_ctw,wemb=0);
    return mods;



def dan_single_model_train_cfg(save_root,dsroot,chcnt,
                               log_path,log_each,itrk= "Top Nep",bsize=160):

    tr_meta_path_ctwch,tr_meta_path_hwdb,te_meta_path_hwdb,te_meta_path_ctw,\
    hwdb_eval_ds,ctw_eval_ds,train_joint_ds=get_dss(dsroot,chcnt,bsize);

    task_dict = {}
    # task_dict = arm_base_mk8_task_default(task_dict, "base_hwdb_",osdanmk8_eval_routine_cfg , 1, te_meta_path_hwdb,hwdb_eval_ds , log_path,force_skip_ctx=True, view_name="synthw", proto_viewname="glyph");
    task_dict = arm_base_mk8_task_default(task_dict, "base_ctw_", osdanmk8_eval_routine_cfg, 1, te_meta_path_ctw, ctw_eval_ds,
                                      log_path,force_skip_ctx=True, view_name="synthw", proto_viewname="glyph");

    routines = {};
    # routines = arm_base_mk8_routine(routines, "base_hwdb_", osdanmk8adt_ocr_routine, 1, log_path,
    #                             log_each, "hwdb_", view_name="synthw", proto_viewname="glyph");
    routines = arm_base_mk8_routine(routines, "base_ctw_", osdanmk8adt_ocr_routine, 1, log_path,
                                  log_each, "ctw_", view_name="synthw", proto_viewname="glyph");

    return \
        {
            "root": save_root,
            "val_each": 2000,
            "vitr": 10000,
            "vepoch": 5,
            "iterkey": itrk,  # something makes no sense to start fresh
            "dataloader_cfg":train_joint_ds,
            # make sure the unseen characters are unseen.
            "modules": model_mod_cfg(tr_meta_path_ctwch,tr_meta_path_hwdb),
            "routine_cfgs": routines,
            "tasks": task_dict,
        }
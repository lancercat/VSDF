from neko_2021_mjt.configs.loadouts.mk8.base_mk8_module_set import arm_base_mk8_routine,arm_base_mk8_task_default
from neko_2021_mjt.configs.routines.ocr_routines.mk8.osdanmk8_routine_cfg import osdanmk8adt_ocr_routine
from neko_2021_mjt.dss_presets.quickviet import get_dss_quickviet; # import preset
from neko_2021_mjt.configs.routines.ocr_routines.mk8.osdanmk8_routine_cfg import  osdanmk8_eval_routine_cfg
from mao_2022_magic.configs.arm_trinorm_module_set_magic import arm_trinorm_mk8hnp_module_set_dan_r45_magic

def model_mod_cfg(tr_meta_path_chs,maxT_viet):
    capacity=256;
    feat_ch=512;
    mods={};
    # Reminder: set ccnt slightly larger than # characters in your data, or it will bite hard.
    mods=arm_trinorm_mk8hnp_module_set_dan_r45_magic(mods,"base_viet_",maxT_viet,capacity,feat_ch,tr_meta_path_chs,ccnt=62,wemb=0);
    return mods;
def dan_single_model_train_cfg(save_root,dsroot,
                               log_path,log_each,itrk= "Top Nep",bsize=48,tvitr=200000):
    maxT_viet=30;
    # load it here
    tr_meta_path_viet, te_meta_path_viet, train_joint_ds, viet_eval_ds=get_dss_quickviet(dsroot,maxT_viet,maxT_viet,bsize);
    task_dict = {}
    task_dict = arm_base_mk8_task_default(task_dict, "base_viet_", osdanmk8_eval_routine_cfg, maxT_viet, te_meta_path_viet, viet_eval_ds,
                                      log_path,force_skip_ctx=True);
    routines = {};
    routines = arm_base_mk8_routine(routines, "base_viet_", osdanmk8adt_ocr_routine, maxT_viet, log_path,
                                     log_each, "dan_viet_", view_name="synthw", proto_viewname="glyph");
    return \
        {
            "root": save_root,
            "val_each": 10000,
            "vitr": 200, # Duh, we have only 3 images
            "vepoch": 3, # 3*200 =60 iters
            "iterkey": itrk,  # something makes no sense to start fresh
            "dataloader_cfg":train_joint_ds,
            # make sure the unseen characters are unseen.
            "modules": model_mod_cfg(tr_meta_path_viet,maxT_viet),
            "routine_cfgs": routines,
            "tasks": task_dict,
        }
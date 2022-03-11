
from neko_2021_mjt.configs.loadouts.mk7.lsctt_module_set_mk7 import arm_module_set_r45trinorm_orig_dsa3hGTAnp_mk7
from neko_2021_mjt.configs.loadouts.base_module_set import arm_base_routine2
from neko_2021_mjt.configs.loadouts.base_module_set import arm_base_task_default2
from neko_2021_mjt.configs.routines.ocr_routines.mk7.osdanmk7_routine_cfg import osdanmk7_eval_routine_cfg
from neko_2021_mjt.dss_presets.dual_no_lsct_32 import get_dss;
from neko_2021_mjt.configs.routines.ocr_routines.mk7.osdanmk7_routine_cfg import  osdanmk7dt_ocr_routine

def model_mod_cfg(tr_meta_path_chs,tr_meta_path_mjst,maxT_mjst,maxT_chs):
    capacity=256;
    feat_ch=512;
    mods={};
    # mods=arm_module_set_r45trinorm_orig_dsa3hGTAnp_mk7(mods,"base_mjst_",maxT_mjst,capacity,feat_ch,tr_meta_path_mjst,wemb=0);
    mods=arm_module_set_r45trinorm_orig_dsa3hGTAnp_mk7(mods,"base_chs_",maxT_chs,capacity,feat_ch,tr_meta_path_chs,wemb=0);
    return mods;



def dan_single_model_train_cfg(save_root,dsroot,
                               log_path,log_each,itrk= "Top Nep",bsize=48,tvitr=200000):
    maxT_chs=30;
    maxT_mjst=25;

    tr_meta_path_chsjap, te_meta_path_chsjap, tr_meta_path_mjst,te_meta_path_mjst, \
    mjst_eval_ds, chs_eval_ds, train_joint_ds=get_dss(dsroot,maxT_mjst,maxT_chs,bsize);

    task_dict = {}
    # task_dict = arm_base_task_default2(task_dict, "base_mjst_", osdanmk7_eval_routine_cfg, maxT_mjst, te_meta_path_mjst,mjst_eval_ds , log_path);
    task_dict = arm_base_task_default2(task_dict, "base_chs_", osdanmk7_eval_routine_cfg, maxT_chs, te_meta_path_chsjap, chs_eval_ds,
                                      log_path);

    routines = {};
    # routines = arm_base_routine2(routines, "base_mjst_", osdanmk7dt_ocr_routine, maxT_mjst, log_path,
    #                             log_each, "dan_mjst_");
    routines = arm_base_routine2(routines, "base_chs_", osdanmk7dt_ocr_routine, maxT_chs, log_path,
                                  log_each, "dan_chs_");

    return \
        {
            "root": save_root,
            "val_each": 10000,
            "vitr": 200000,
            "vepoch": 2,
            "iterkey": itrk,  # something makes no sense to start fresh
            "dataloader_cfg":train_joint_ds,
            # make sure the unseen characters are unseen.
            "modules": model_mod_cfg(tr_meta_path_chsjap,tr_meta_path_mjst, maxT_mjst,maxT_chs),
            "routine_cfgs": routines,
            "tasks": task_dict,
        }
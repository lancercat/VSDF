from neko_2021_mjt.routines.ocr_routines.mk8.osdan_routine_mk8 import neko_HDOS2C_routine_CFmk8,\
    neko_HDOS2C_eval_routine_CFmk8,neko_HDOS2C_routine_CFmk8a,neko_HDOS2C_routine_CFmk8adt
def osdanmk8_ocr_routine(sampler_name,prototyper_name,semantic_branch_name,feature_extractor_name,seq_name,
                         TA_name,pred_name,ctxmodule_name,loss_name,ctxloss_name,label_name,image_name,mask_name,log_path,log_each,name,maxT):
    return \
    {

        "maxT": maxT,
        "name":name,
        "routine":neko_HDOS2C_routine_CFmk8,
        "mod_cvt_dicts":
        {
            "semantic_branch":semantic_branch_name,
            "sampler": sampler_name,
            "prototyper":prototyper_name,
            "feature_extractor":feature_extractor_name,
            "TA":TA_name,
            "seq": seq_name,
            "preds":pred_name,
            "ctxmodule":ctxmodule_name,
            "losses":loss_name,
            "ctxloss":ctxloss_name,
        },
        "inp_cvt_dicts":
        {
            "label":label_name,
            "image":image_name,
            "mask":mask_name,
        },
        "log_path":log_path,
        "log_each":log_each,
    };
def osdanmk8a_ocr_routine(sampler_name,prototyper_name,semantic_branch_name,feature_extractor_name,seq_name,
                         TA_name,pred_name,ctxmodule_name,loss_name,ctxloss_name,label_name,image_name,mask_name,log_path,log_each,name,maxT):
    return \
    {

        "maxT": maxT,
        "name":name,
        "routine":neko_HDOS2C_routine_CFmk8a,
        "mod_cvt_dicts":
        {
            "semantic_branch":semantic_branch_name,
            "sampler": sampler_name,
            "prototyper":prototyper_name,
            "feature_extractor":feature_extractor_name,
            "TA":TA_name,
            "seq": seq_name,
            "preds":pred_name,
            "ctxmodule":ctxmodule_name,
            "losses":loss_name,
            "ctxloss":ctxloss_name,
        },
        "inp_cvt_dicts":
        {
            "label":label_name,
            "image":image_name,
            "mask":mask_name,
        },
        "log_path":log_path,
        "log_each":log_each,
    };

def osdanmk8adt_ocr_routine(sampler_name,prototyper_name,semantic_branch_name,feature_extractor_name,seq_name,
                         TA_name,pred_name,ctxmodule_name,loss_name,ctxloss_name,label_name,image_name,mask_name,log_path,log_each,name,maxT):
    return \
    {

        "maxT": maxT,
        "name":name,
        "routine":neko_HDOS2C_routine_CFmk8adt,
        "mod_cvt_dicts":
        {
            "semantic_branch":semantic_branch_name,
            "sampler": sampler_name,
            "prototyper":prototyper_name,
            "feature_extractor":feature_extractor_name,
            "TA":TA_name,
            "seq": seq_name,
            "preds":pred_name,
            "ctxmodule":ctxmodule_name,
            "losses":loss_name,
            "ctxloss":ctxloss_name,
        },
        "inp_cvt_dicts":
        {
            "label":label_name,
            "image":image_name,
            "mask":mask_name,
        },
        "log_path":log_path,
        "log_each":log_each,
    };

def osdanmk8_eval_routine_cfg(sampler_name,prototyper_name,semantic_branch_name,feature_extractor_name,
                         TA_name,seq_name,pred_name,ctxmodule_name,loss_name,image_name,label_name,mask_name,log_path,name,maxT,measure_rej=False):
    return \
    {
        "name":name,
        "maxT": maxT,
        "routine":neko_HDOS2C_eval_routine_CFmk8,
        "mod_cvt_dicts":
        {
            "sampler":sampler_name,
            "prototyper":prototyper_name,
            "semantic_branch": semantic_branch_name,
            "feature_extractor":feature_extractor_name,
            "TA":TA_name,
            "seq": seq_name,
            "preds":pred_name,
            "ctxmodule": ctxmodule_name,
            "losses":loss_name,
        },
        "inp_cvt_dicts":
        {
            "label": label_name,
            "image": image_name,
            "mask":mask_name,
            "proto": "proto",
            "plabel": "plabel",
            "fsp": "fsp",
            "csp": "csp",
            "tdict": "tdict",
            "gtdict": "gtdict",
        },
        "log_path":log_path,
        "measure_rej":measure_rej,
    };
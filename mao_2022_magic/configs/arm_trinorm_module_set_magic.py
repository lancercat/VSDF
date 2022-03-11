
from neko_2021_mjt.configs.common_subs.arm_postfe import arm_rest_commonwops
from neko_2021_mjt.configs.modules.config_cls_emb_loss import\
    config_cls_emb_loss2,config_cls_lossohem
from neko_2021_mjt.configs.loadouts.mk8.base_mk8_module_set import arm_trinorm_mk8h_common,arm_mk8_proto_sganp

from mao_2022_magic.configs.mao_magic_semantic_branch import config_ctxmodule

def arm_mk8_ctx_magic(srcdst,prefix,feat_ch,nhead=8,nlay=4):
    srcdst[prefix+"ctxloss"]=  config_cls_emb_loss2();
    srcdst[prefix+"ctxmodule"]=config_ctxmodule(feat_ch,nhead,nlay);
    return srcdst;

def arm_trinorm_mk8hnp_module_set_dan_r45_magic(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1,views=["synthw","glyph"],ccnt=3900,camch=64,sheads=8,slays=4,wemb=0.3):
    srcdst=arm_trinorm_mk8h_common(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=expf,views=views,ccnt=ccnt,camch=camch);
    srcdst=arm_mk8_proto_sganp(srcdst,prefix,capacity,feat_ch);
    srcdst = arm_rest_commonwops(srcdst,prefix,wemb=wemb);
    srcdst=arm_mk8_ctx_magic(srcdst,prefix,feat_ch,sheads,slays);
    return srcdst;


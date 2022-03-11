from neko_2021_mjt.dataloaders.neko_joint_loader import neko_joint_loader
from neko_2021_mjt.configs.data.mjst_data import get_mjstcqa_cfg,get_test_all_uncased_dsrgb
from neko_2021_mjt.eval_tasks.dan_eval_tasks import neko_odan_eval_tasks
from neko_2021_mjt.configs.data.chs_jap_data import get_chs_HScqa,get_eval_jap_color,get_chs_tr_meta64,get_jap_te_meta64
import os

def get_dataloadercfgs(root,te_meta_path,tr_meta_path,maxT_mjst,maxT_chsHS,bsize,random_aug):
    return \
    {
        "loadertype": neko_joint_loader,
        "subsets":
        {
            # reset iterator gives deadlock, so we give a large enough repeat number
            "dan_mjst":get_mjstcqa_cfg(root, maxT_mjst, bs=bsize,hw=[32,128],random_aug=random_aug),
        }
    }

def get_dss(dsroot,maxT_mjst,maxT_chs,bsize,random_aug=True):

    tr_meta_path_chsjap = get_chs_tr_meta64(dsroot);
    te_meta_path_chsjap = get_jap_te_meta64(dsroot);
    tr_meta_path_mjst = os.path.join(dsroot, "dicts", "dab62cased64.pt");
    te_meta_path_mjst = os.path.join(dsroot, "dicts", "dab62cased64.pt");
    mjst_eval_ds = get_test_all_uncased_dsrgb(maxT_mjst, dsroot, None, 32,hw=[32,128])
    train_joint_ds=get_dataloadercfgs(dsroot,te_meta_path_mjst,tr_meta_path_mjst,maxT_mjst,maxT_chs,bsize,random_aug);
    return tr_meta_path_chsjap,te_meta_path_chsjap,tr_meta_path_mjst,te_meta_path_mjst,mjst_eval_ds,None,train_joint_ds
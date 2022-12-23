from neko_2021_mjt.dataloaders.neko_joint_loader import neko_joint_loader
from neko_2021_mjt.configs.data.quickviet_data import get_quickviet_training_cfg,get_test_quickviet_uncased_dsrgb


def get_dataloadercfgs(root,te_meta_path,tr_meta_path,maxT_mjst,maxT_chsHS,bsize,random_aug):
    return \
    {
        "loadertype": neko_joint_loader,
        "subsets":
        {
            # reset iterator gives deadlock, so we give a large enough repeat number
            "dan_viet":get_quickviet_training_cfg(root, maxT_mjst, bs=bsize,hw=[32,128],random_aug=random_aug),
        }
    }

def get_dss_quickviet(dsroot,maxT_mjst,maxT_chs,bsize,random_aug=True):
    tr_meta_path_viet = "/run/media/lasercat/writebuffer/quickviet/meta/dict.pt";
    te_meta_path_viet = "/run/media/lasercat/writebuffer/quickviet/meta/dict.pt";
    # duh I am just demonstrating how to train and test with new data, so I am using one dataset for both training and testing,
    viet_eval_ds = get_test_quickviet_uncased_dsrgb(maxT_mjst, dsroot, None, 32,hw=[32,128]);
    train_joint_ds=get_dataloadercfgs(dsroot,None,None,maxT_mjst,maxT_chs,bsize,random_aug);
    return tr_meta_path_viet,te_meta_path_viet,train_joint_ds,viet_eval_ds;
from neko_2021_mjt.dataloaders.neko_joint_loader import neko_joint_loader
from neko_2021_mjt.configs.data.orahw_data_fsl import get_eval_orahw,get_orahw_train,get_orahw_te_meta,get_orahw_tr_meta

def get_dataloadercfgs(root):
    return \
    {
        "loadertype": neko_joint_loader,
        "subsets":
        {
            # reset iterator gives deadlock, so we give a large enough repeat number
            "ofsl_orahw":get_orahw_train(root),
        }
    }

def get_eval_dss(dsroot):
    te_meta_path = get_orahw_te_meta(dsroot);
    eval_ds = get_eval_orahw(dsroot);
    return te_meta_path,eval_ds
def get_dss(dsroot):
    te_meta_path,eval_ds=get_eval_dss(dsroot);
    train_joint_ds=get_dataloadercfgs(dsroot);
    return train_joint_ds,te_meta_path,eval_ds;

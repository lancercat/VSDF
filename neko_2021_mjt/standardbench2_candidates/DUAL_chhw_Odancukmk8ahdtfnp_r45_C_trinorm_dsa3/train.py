from neko_2021_mjt.neko_abstract_jtr import neko_abstract_modular_joint_training,neko_modular_joint_training_para
from configs import dan_single_model_train_cfg
from neko_sdk.root import find_data_root;

import torch
import sys

if __name__ == '__main__':
    if(len(sys.argv)>1):
        ccnt=int(sys.argv[1]);
    else:
        ccnt=500

    trainer=neko_modular_joint_training_para(
        dan_single_model_train_cfg(
            "jtrmodels"+str(ccnt),
            find_data_root(),
            ccnt,
            "../logs/",
            200,
        )
    );
    # with torch.autograd.detect_anomaly():
    trainer.train(None);

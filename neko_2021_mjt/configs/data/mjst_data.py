
from neko_2020nocr.dan.dataloaders.dataset_scene import colored_lmdbDataset,colored_lmdbDatasetT
from neko_2020nocr.dan.configs.datasets.ds_paths import *
from torchvision import transforms
from torch.utils.data import DataLoader


def get_mjstcqa_cfg(root,maxT,bs=48,hw=[32,128],random_aug=True):
    rdic={
                "type": colored_lmdbDataset,
                'ds_args': {
                    'roots': [get_nips14(root), get_cvpr16(root)],
                    'img_height': hw[0],
                    'img_width': hw[1],
                    'transform': transforms.Compose([transforms.ToTensor()]),
                    'global_state': 'Train',
                    "maxT": maxT,
                    'qhb_aug': random_aug
                },
                "dl_args":
                {
                    'batch_size': bs,
                    'shuffle': False,
                    'num_workers': 8,
                }
            }
    return rdic;



def get_dataset_testC(maxT,root,dict_dir,batch_size=128,hw=[32,128]):
    return {
        'type': colored_lmdbDatasetT,
        'ds_args': {
            'roots': [root],
            'img_height': hw[0],
            'img_width': hw[1],
            'transform': transforms.Compose([transforms.ToTensor()]),
            'global_state': 'Test',
            "maxT": maxT,
        },
        'dl_args': {
            'batch_size': batch_size,
            'shuffle': False,
            'num_workers': 8,
        },
    }
def get_test_all_uncased_dsrgb(maxT=25,root="/home/lasercat/ssddata/",dict_dir='../../dict/dic_36.txt',batchsize=128,hw=[32,128]):
    return {
        "dict_dir":dict_dir,
        "case_sensitive": False,
        "te_case_sensitive": False,
            "datasets":{
            "CUTE": get_dataset_testC(maxT, get_cute(root), dict_dir,batchsize,hw),
            "IIIT5k": get_dataset_testC(maxT, get_iiit5k(root), dict_dir,batchsize,hw),
            "SVT": get_dataset_testC(maxT, get_SVT(root), dict_dir,batchsize,hw),
            "IC03": get_dataset_testC(maxT, get_IC03_867(root), dict_dir,batchsize,hw),
            "IC13": get_dataset_testC(maxT, get_IC13_1015(root), dict_dir,batchsize,hw),
            }
        }

def get_test_synth_train_uncased_dsrgb(maxT=25,root="/home/lasercat/ssddata/",dict_dir='../../dict/dic_36.txt',batchsize=128,hw=[32,128]):
    return {
        "dict_dir":dict_dir,
        "case_sensitive": False,
        "te_case_sensitive": False,
            "datasets":{
            "NIPS": get_dataset_testC(maxT, get_nips14sub(root), dict_dir,batchsize,hw),
            "CVPR": get_dataset_testC(maxT, get_cvpr16sub(root), dict_dir,batchsize,hw),
            }
        }
def get_uncased_dsrgb_d_tr(maxT=25,root="/home/lasercat/ssddata/",dict_dir='../../dict/dic_36.txt',batchsize=128):
    return {
        "dict_dir":dict_dir,
        "case_sensitive": False,
        "te_case_sensitive": False,
            "datasets":{
            }
        }
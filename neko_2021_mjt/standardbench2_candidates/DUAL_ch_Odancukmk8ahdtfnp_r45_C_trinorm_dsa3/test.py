from eval_configs import dan_mjst_eval_cfg
from neko_sdk.root import find_export_root,find_model_root

if __name__ == '__main__':
    import sys
    for i in ["500","1000","1500","2000"]:
        print("-----------------",i,"starts");
        if(len(sys.argv)<2):
            argv = ["Meeeeooooowwww",
                    None,#find_export_root()+"/DUAL_ch_Odancukmk8ahdtfnp_r45_C_trinorm_dsa3/jtrmodels"+i+"/",
                    "_E4",
                    find_model_root()+"/DUAL_ch_Odancukmk8ahdtfnp_r45_C_trinorm_dsa3/jtrmodels"+i+"/",
                    ]
        else:
            argv=sys.argv;
        from neko_2021_mjt.lanuch_std_test import launchtest
        launchtest(argv,dan_mjst_eval_cfg)
        print("-----------------",i,"ends");

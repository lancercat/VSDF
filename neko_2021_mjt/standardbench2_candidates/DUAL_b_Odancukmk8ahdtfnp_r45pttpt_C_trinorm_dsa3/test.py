from eval_configs import dan_mjst_eval_cfg
from neko_sdk.root import find_export_root,find_model_root

if __name__ == '__main__':
    import sys
    if(len(sys.argv)<2):
        argv = ["Meeeeooooowwww",
                None,#,find_export_root()+"/DUAL_b_Odancukmk8ahdtfnp_r45pttpt_C_trinorm_dsa3/jtrmodels",
                "_E3",
                find_model_root()+"/DUAL_b_Odancukmk8ahdtfnp_r45pttpt_C_trinorm_dsa3/jtrmodels",
                ]
    else:
        argv=sys.argv;
    from neko_2021_mjt.lanuch_std_test import launchtest
    launchtest(argv,dan_mjst_eval_cfg)

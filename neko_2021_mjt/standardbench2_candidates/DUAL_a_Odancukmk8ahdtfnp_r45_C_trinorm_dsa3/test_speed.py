from eval_configs import dan_mjst_eval_cfg

if __name__ == '__main__':
    import sys
    if(len(sys.argv)<2):
        argv = ["Meeeeooooowwww",
                None,
                "_E1",
                "/run/media/lasercat/ssddata/cvpr22_candidata/8ahdt/",
                ]
    else:
        argv=sys.argv;
    from neko_2021_mjt.lanuch_std_test import launchtest
    launchtest(argv,dan_mjst_eval_cfg)

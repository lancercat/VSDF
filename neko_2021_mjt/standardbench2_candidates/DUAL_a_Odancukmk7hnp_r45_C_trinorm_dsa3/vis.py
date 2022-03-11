from vis_configs import dan_mjst_eval_cfg
from neko_2021_mjt.lanuch_flashtorch import launchflashtorch

if __name__ == '__main__':
    import sys
    if(len(sys.argv)<2):
        argv = ["Meeeeooooowwww",
                "/run/media/lasercat/ssddata/cvpr22_candidata/7h/",
                "_E1",
                "/run/media/lasercat/ssddata/cvpr22_candidata/7h/",
                ]
    else:
        argv=sys.argv;
    launchflashtorch(argv,dan_mjst_eval_cfg)

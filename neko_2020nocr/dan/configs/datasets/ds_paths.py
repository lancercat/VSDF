import os;
def get_nips14(root="/home/lasercat/ssddata/"):
    return os.path.join(root,'NIPS2014');
def get_cvpr16(root="/home/lasercat/ssddata/"):
    return os.path.join(root,'CVPR2016');
def get_nips14sub(root="/home/lasercat/ssddata/"):
    return os.path.join(root,'NIPS2014sub');
def get_cvpr16sub(root="/home/lasercat/ssddata/"):
    return os.path.join(root,'CVPR2016sub');
def get_iiit5k(root="/home/lasercat/ssddata/"):
    return os.path.join(root,'IIIT5k_3000');
def get_cute(root="/home/lasercat/ssddata/"):
    return os.path.join(root,'CUTE80');
def get_IC03_867(root="/home/lasercat/ssddata/"):
    return os.path.join(root,'IC03_867');
def get_IC13_1015(root="/home/lasercat/ssddata/"):
    return os.path.join(root,"IC13_1015")
def get_IC15_2077(root="/home/lasercat/ssddata/"):
    return os.path.join(root,'IC15_2077');
def get_IC15_1811(root="/home/lasercat/ssddata/"):
    return os.path.join(root,'IC15_1811');
def get_SVT(root="/home/lasercat/ssddata/"):
    return os.path.join(root,'SVT');



def get_lsvt_path(root="/home/lasercat/ssddata/"):
    return os.path.join(root,'lsvtdb');
def get_ctw(root="/home/lasercat/ssddata/"):
    return os.path.join(root,'ctw_fslchr');
def get_art_path(root="/home/lasercat/ssddata/"):
    return os.path.join(root,'arttrdb');


def get_ctw2k(root="/home/lasercat/ssddata/"):
    return os.path.join(root,"ctw2kseen")

def get_ctw2kus(root="/home/lasercat/ssddata/"):
    return os.path.join(root,"ctw2kunseen_500")

def get_artK_path(root="/home/lasercat/ssddata/"):
    return os.path.join(root,'arttrdbseen');
def get_mlt_chlatK_path(root="/home/lasercat/ssddata/"):
    return os.path.join(root,'mltchlatdbseen');
def get_lsvtK_path(root="/home/lasercat/ssddata/"):
    return os.path.join(root,'lsvtdbseen');
def get_ctwK_path(root="/home/lasercat/ssddata/"):
    return os.path.join(root,'ctw_seen');
def get_rctwK_path(root="/home/lasercat/ssddata/"):
    return os.path.join(root,'rctwdb_seen');


def get_mlt_chlat_path(root="/home/lasercat/ssddata/"):
    return os.path.join(root,'mltchlatdb');
def get_mlt_chlatval(root="/home/lasercat/ssddata/"):
    return os.path.join(root,'mltchlatdbval');
def get_artvalseen(root="/home/lasercat/ssddata/"):
    return os.path.join(root,'arttrvaldbseen');
def get_artval(root="/home/lasercat/ssddata/"):
    return os.path.join(root,'arttrvaldb');
def get_8mtr(root="/home/lasercat/ssddata/"):
    ds=[
        os.path.join(root,'8mtr_1'),
        os.path.join(root,'8mtr_2'),
        os.path.join(root,'8mtr_3'),
    ]
    return ds;
def get_qhbcsvtr(root="/home/lasercat/ssddata/"):
    return os.path.join(root,'qhbcsvtr');


def get_docuevalseen(root="/home/lasercat/ssddata/"):
    return os.path.join(root,'docstchsevalseen');
def get_mlt_artvalunseen(root="/home/lasercat/ssddata/"):
    return os.path.join(root,'arttrvaldbunseen');
def get_mlt_jpval(root="/home/lasercat/ssddata/"):
    return os.path.join(root,'mltvaljpdb');
def get_ulsvta(root="/home/lasercat/ssddata/"):
    return os.path.join(root,'lsvtdbunseenA');

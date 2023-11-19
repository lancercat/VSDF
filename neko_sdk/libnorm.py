import numpy as np;
import cv2;
import math;
def distance(src,dst):
    td=0;
    for i in range(4):
        dv=(src[i]-dst[i]);
        d=dv[0]*dv[0]+dv[1]*dv[1];
        d=math.sqrt(d);
        td+=d;
    return td;

def mapping(src,dst):
    best=0xca71*0xca39;
    best_mapping=9;
    for i in range(4):
        d=distance(src,dst);
        if(d<best):
            best=d;
            best_mapping=src.copy();
        src=np.roll(src, 1, axis=0);
    return best_mapping,best;

def norm_sz(cords):
    src_cords = np.float32(cords);
    rect = cv2.minAreaRect(src_cords);
    le=max(rect[1][0], rect[1][1]);
    se=min(rect[1][0], rect[1][1]);
    v=cords[:, 1].max()-cords[:, 1].min()
    h=cords[:, 0].max()-cords[:, 0].min()
    if(h*1.2>=v):
        sz = (le,se);
    else:
        sz =(se,le);
    return sz,np.array([rect[0][0],rect[0][1]]);

def norm(src,cords,size,ratio=1.1):
    # rect = cv2.minAreaRect(cords);
    # new_rect=(rect[0],(rect[1][0],rect[1][1]),rect[2]);
    # cords=cv2.boxPoints(new_rect);
    src_cords = np.float32(cords);
    if(size is None):
        sz,ctr=norm_sz(cords);
    else:
        sz,ctr=size,src_cords.mean(axis=0);
    dst_cords_h=np.float32([[0,0],[sz[0],0],[sz[0],sz[1]],[0,sz[1]]]);
    dst_cords_v = np.float32([[0, 0], [sz[1], 0], [sz[1], sz[0]], [0, sz[0]]]);

    src_cordsh,dh=mapping(src_cords,dst_cords_h-np.array([sz[0]/2,sz[1]/2])+ctr);
    src_cordsv,dv=mapping(src_cords,dst_cords_v-np.array([sz[1]/2,sz[0]/2])+ctr);
    if(dh>dv):
        src_cords=src_cordsv;
        dst_cords=dst_cords_v;
        sz=(sz[1],sz[0]);
    else:
        src_cords=src_cordsh;
        dst_cords=dst_cords_h;

    M=cv2.getPerspectiveTransform(src_cords,dst_cords);
    dst=cv2.warpPerspective(src,M,(int(sz[0]),int(sz[1])));
    return dst;
def normrect_with_mask(src,gmask,cords):
    lmask=np.zeros([src.shape[0],src.shape[1]],dtype=np.uint8);
    cv2.fillPoly(lmask, [cords],255);
    cv2.imshow("tmp",lmask);
    cv2.waitKey(100);
    a,b,c,d=cords[:,1].min(),cords[:,1].max(),cords[:,0].min(),cords[:,0].max();
    return src[a:b,c:d],lmask[a:b,c:d],gmask[a:b,c:d];


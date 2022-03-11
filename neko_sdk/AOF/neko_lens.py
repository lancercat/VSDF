from torch import nn;
import torch;
import torch.nn.functional as trnf;


class neko_dense_calc(nn.Module):
    def __init__(this,channel):
        super(neko_dense_calc, this).__init__();
        ## 996-251. I am not on purpose.
        ## The sigmoid perserves topology.
        this.dkern=torch.nn.Sequential(
            nn.Conv2d(int(channel),2,5,1),
            torch.nn.Sigmoid());
    def forward(this,feat,lw,lh):
        if(this.training):
            dmap = this.dkern(feat/(feat.norm(dim=1,keepdim=True)+0.00000009));
        else:
            norm=feat.norm(dim=1,keepdim=True);
            dmap = this.dkern(feat/(norm+0.00000009)*(norm>0.09));
            # During large-batch evaluation, numeric errors seems to get much larger than eps,
            # causing sever performance loss, hence we commence this hot-fix.

        ndmap = trnf.interpolate(dmap, [lh + 2, lw + 2]);
        return ndmap;

class neko_dense_calcnn(nn.Module):
    def __init__(this,channel):
        super(neko_dense_calcnn, this).__init__();
        ## 996-251. I am not on purpose.
        ## The sigmoid perserves topology.
        this.dkern=torch.nn.Sequential(
            nn.Conv2d(int(channel),2,5,1),
            torch.nn.Sigmoid());
    def forward(this,feat,lw,lh):
        dmap = this.dkern(feat);
        ndmap = trnf.interpolate(dmap, [lh + 2, lw + 2]);
        return ndmap;



def neko_dense_norm(ndmap):
    [h__, w__] = ndmap.split([1, 1], 1);
    sumh=torch.sum(h__, dim=2, keepdim=True);
    sumw=torch.sum(w__,dim=3,keepdim=True);
    h_ = h__ / sumh;
    w_ = w__ / sumw;
    h = torch.cumsum(h_, dim=2);
    w = torch.cumsum(w_, dim=3);
    nidx = torch.cat([ w[:, :, 1:-1, 1:-1],h[:, :, 1:-1, 1:-1]], dim=1)* 2 - 1;
    return nidx;

def neko_sample(feat,grid,dw,dh):
    dst = trnf.grid_sample(feat, grid.permute(0, 2, 3, 1),mode="bilinear");
    return trnf.adaptive_avg_pool2d(dst,[dh,dw]);

def neko_dsample(feat,dmap,dw,dh):
    grid = neko_dense_norm(dmap);
    dst = neko_sample(feat, grid, dw, dh);
    return dst
def neko_idsample(feat,density,dw,dh):
    rdense=1/density;
    grid = neko_dense_norm(rdense);
    dst = neko_sample(feat, grid, dw, dh);
    return dst;

def neko_dense_norm2(ndmap):
    [h__, w__] = ndmap.split([1, 1], 1);
    sumh=torch.sum(h__, dim=2, keepdim=True);
    sumw=torch.sum(w__,dim=3,keepdim=True);
    h_ = h__ / sumh;
    w_ = w__ / sumw;
    h = torch.cumsum(h_, dim=2);
    w = torch.cumsum(w_, dim=3);
    nidx = torch.cat([ w,h], dim=1)* 2 - 1;
    return nidx;
def neko_sample2(feat,grid):
    dst = trnf.grid_sample(feat, grid.permute(0, 2, 3, 1),mode="bilinear",align_corners=True);
    return dst;

def neko_dsample2(feat,dmap):
    grid = neko_dense_norm(dmap);
    dst = neko_sample2(feat, grid);
    return dst
def neko_idsample2(feat,density):
    rdense=1/density;
    grid = neko_dense_norm(rdense);
    dst = neko_sample2(feat, grid);
    return dst;

def vis_lenses(img,lenses):
    oups=[img];
    for lens in lenses:
        dmap=trnf.interpolate(lens, [img.shape[-2],img.shape[-1]])
        grid=neko_dense_norm(dmap);
        img=neko_sample(img,grid,img.shape[3],img.shape[2])
        oups.append(img);
    return oups;

class neko_lens(nn.Module):
    DENSE=neko_dense_calc
    def __init__(this,channel,pw,ph,hardness=2,dbg=False):
        super(neko_lens, this).__init__();

        this.pw=pw;
        this.ph=ph;
        this.dbg=dbg;
        this.hardness=hardness;
        this.dkern=this.DENSE(channel)


    def forward(this,feat):
        dw = feat.shape[3] // this.pw;
        dh = feat.shape[2] // this.ph;
        dmap=this.dkern(feat,dw,dh);
        dmap += 0.0009;  # sigmoid can be clipped to 0!
        dmap[:, :, 1:-1, 1:-1] += this.hardness;
        dst=neko_dsample(feat,dmap,dw,dh);

        if(not this.dbg):
            return dst,dmap.detach();
        else:
            return dst,dmap.detach();
class neko_lensnn(neko_lens):
    DENSE=neko_dense_calcnn

class neko_lens_w_mask(nn.Module):
    def __init__(this,channel,pw,ph,hardness=2,dbg=False):
        super(neko_lens_w_mask, this).__init__();

        this.pw=pw;
        this.ph=ph;
        this.dbg=dbg;
        this.hardness=hardness;
        this.dkern=neko_dense_calc(channel)


    def forward(this,feat,mask):
        dw = feat.shape[3] // this.pw;
        dh = feat.shape[2] // this.ph;
        dmap=this.dkern(feat,dw,dh);
        dmap += 0.0009;  # sigmoid can be clipped to 0!
        dmap[:, :, 1:-1, 1:-1] += this.hardness;

        grid=neko_dense_norm(dmap);
        dst=neko_sample(feat,grid,dw,dh);
        with torch.no_grad():
            dmsk=neko_sample(mask,grid,dw,dh);
        if(not this.dbg):
            return dst,dmsk,dmap.detach();
        else:
            return dst,dmsk,dmap.detach();

class neko_lens_self(nn.Module):
    def __init__(this,ich,channel,pw,ph,hardness=2,dbg=False,scale=None):
        super(neko_lens_self, this).__init__();
        this.scale=scale;
        this.pw=pw;
        this.ph=ph;
        this.dbg=dbg;
        if(type(hardness) is not tuple):
            this.hardness=hardness;
        else:
            this.hardness=nn.Parameter(torch.tensor(hardness).reshape(1,-1,1,1),requires_grad=False);
        this.ekern=torch.nn.Sequential(
            nn.Conv2d(ich,channel,3,1,1),
            nn.BatchNorm2d(channel),
            nn.ReLU6()
        )
        this.dkern=neko_dense_calc(channel)

    def forward_dsz(this,feat,dsz):
        dw = feat.shape[3] // this.pw;
        dh = feat.shape[2] // this.ph;
        lfeat = this.ekern(feat)
        dmap = this.dkern(lfeat, dw, dh);
        dmap += 0.0009;  # sigmoid can be clipped to 0!
        dmap[:, :, 1:-1, 1:-1] += this.hardness;
        # this.scale=0.25
        if (dsz is not None):
            dsm = trnf.interpolate(dmap,dsz, mode="bilinear");
            dmap = trnf.interpolate(dsm, [dmap.shape[2], dmap.shape[3]], mode="bilinear");

        grid = neko_dense_norm(dmap);

        dst = neko_sample(feat, grid, dw, dh);
        if (not this.dbg):
            return dst, dmap.detach();
        else:
            return dst, dmap.detach();

    def forward(this,feat):
        dsz=[int(feat.shape[2]*this.scale[0]),int(feat.shape[3]*this.scale[1])];
        return this.forward_dsz(feat,dsz);
class neko_lens_self_bogotps(neko_lens_self):
    def forward(this, feat):
        return this.forward_dsz(feat, this.scale);



class neko_lens_fuse(nn.Module):
    def __init__(this,channel,hardness=0.5,dbg=False):
        super(neko_lens_fuse, this).__init__();
        this.hidense=neko_dense_calc(channel);
        this.lodense=neko_dense_calc(channel);
        # allow it to oversample


        this.dbg=dbg;
        this.hardness=hardness;


    def forward(this,hifeat,lofeat):
        dw = lofeat.shape[3];
        dh = lofeat.shape[2];
        lw=hifeat.shape[3];
        lh = hifeat.shape[2];
        hidmap=this.hidense(hifeat,lw,lh);
        lodmap=this.lodense(lofeat,lw,lh);
        dmap=(hidmap+lodmap)/2;
        dmap += 0.0009;  # sigmoid can be clipped to 0!
        dmap[:, :, 1:-1, 1:-1] += this.hardness;
        grid=neko_dense_norm(dmap);
        his=neko_sample(hifeat,grid,dw,dh);
        los=neko_sample(lofeat,grid,dw,dh);
        return his+los,dmap.detach();

if __name__ == '__main__':
    import cv2;
    import numpy as np;
    im=cv2.imread("/home/lasercat/Downloads/nos1080.png");
    w,h=im.shape[1],im.shape[0];
    tim=torch.tensor(im).float().permute(2,0,1).unsqueeze(0);
    dense=torch.rand([1,2,16,32])+0.2;
    dense=torch.nn.functional.interpolate(dense,[h,w],mode="bilinear");
    dense[:,1:]=1;
    jim=    neko_dsample(tim,dense,w,h);
    rjim=   neko_idsample(jim,dense,w,h);
    jim=    jim[0].permute(1,2,0).detach().cpu().numpy().astype(np.uint8);
    rjim = rjim[0].permute(1,2,0).detach().cpu().numpy().astype(np.uint8);
    cv2.imshow("orig",im);
    cv2.imshow("jit", jim);
    cv2.imshow("rjim", rjim);
    cv2.waitKey(0);


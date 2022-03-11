# bogomodules are certain combination to the modules, they do not hold parameters
# instead they use whatever armed to the module set.
# Different to routines, they are statically associated to a certain set of modules for speed up.
import torch;
from torch.nn import functional as trnf
# some time you cannot have a container. That's life.
# say u have module a b c d.
# A uses [ac] B uses [ab] C uses [ad]...
# There is no better way than simply put a,b,c,d in a big basket.

class prototyper_gen3:
    def __init__(this,args,moddict):
        this.force_proto_shape=args["force_proto_shape"];
        this.capacity=args["capacity"];
        # sometimes we do not need sp protos.
        if(args["sp_proto"] in moddict):
            this.sp=moddict[args["sp_proto"]]
        else:
            this.sp=None;
        this.backbone=moddict[args["backbone"]];
        this.cam=moddict[args["cam"]];
        this.device_indicator="cuda";
        if(args["drop"] is not None):
            this.drop=moddict[args["drop"]];
        else:
            this.drop=None;
    def freeze(this):
        this.backbone.model.freeze();
        if(this.sp is not None):
            this.sp.eval();
        this.cam.eval();

    def freeze_bb_bn(this):
        this.backbone.model.freezebn();

    def unfreeze_bb_bn(this):
        this.backbone.model.unfreezebn();


    def unfreeze(this):
        this.backbone.model.unfreeze();
        if(this.sp is not None):
            this.sp.train();
        this.cam.train();

    def cuda(this):
        this.device_indicator="cuda";
    def proto_engine(this,clips):
        features = this.backbone(clips)
        A = this.cam(features);
        # A=torch.ones_like(A);
        out_emb=(A*features[-1]).sum(-1).sum(-1)/A.sum(-1).sum(-1);
        return out_emb;

    def forward(this,normprotos,rot=0,use_sp=True):
        if (len(normprotos) <= this.capacity):
            # pimage=torch.cat(normprotos).to(this.dev_ind.device);
            pimage = torch.cat(normprotos).contiguous().to(this.device_indicator);
            if(this.force_proto_shape is not None and pimage.shape[-1] !=this.force_proto_shape):
                pimage=trnf.interpolate(pimage,[this.force_proto_shape,this.force_proto_shape],mode="bilinear");
            if (rot > 0):
                pimage = torch.rot90(pimage, rot, [2, 3]);

            if (pimage.shape[1] == 1):
                pimage = pimage.repeat([1, 3, 1, 1]);
            if(use_sp):
                spproto, _ = this.sp();
                proto = [spproto, this.proto_engine(pimage)];
            else:
                proto=[this.proto_engine(pimage)]
        else:
            if(use_sp):
                spproto, _ = this.sp();
                proto = [spproto];
            else:
                proto=[];
            chunk=this.capacity//4;
            for s in range(0, len(normprotos),chunk ):
                pimage = torch.cat(normprotos[s:s + chunk]).contiguous().to(this.device_indicator);
                if (rot > 0):
                    pimage = torch.rot90(pimage, rot, [2, 3]);
                if (pimage.shape[1] == 1):
                    pimage = pimage.repeat([1, 3, 1, 1]);
                    proto.append(this.proto_engine(pimage))
        allproto = trnf.normalize(torch.cat(proto), dim=1, eps=0.0009);
        if (this.drop):
            allproto = this.drop(allproto);
        return allproto.contiguous();
    def __call__(this, *args, **kwargs):
        return this.forward(*args,**kwargs);
class prototyper_gen3d(prototyper_gen3):
    def proto_engine(this,clips):
        features = this.backbone(clips)
        A = this.cam([f.detach() for f in features]);
        # A=torch.ones_like(A);
        out_emb=(A*features[-1]).sum(-1).sum(-1)/A.sum(-1).sum(-1);
        return out_emb;

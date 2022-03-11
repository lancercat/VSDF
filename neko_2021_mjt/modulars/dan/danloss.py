
from torch import nn
import torch
import torch_scatter
from torch.nn import functional as trnf
try:
    import pylcs
except:
    pylcs=None;
    print("no pylcs!, some loss (trident net) won't work!");

class osdanloss(nn.Module):
    def __init__(this, cfgs):
        super(osdanloss, this).__init__();
        this.setuploss(cfgs);

    def label_weight(this, shape, label):
        weight = torch.zeros(shape).to(label.device) + 0.1
        weight = torch.scatter_add(weight, 0, label, torch.ones_like(label).float());
        weight = 1. / weight;
        weight[-1] /= 200;
        return weight;

    def setuploss(this, cfgs):
        # this.aceloss=
        this.cosloss = neko_cos_loss2().cuda();
        this.wcls = cfgs["wcls"];
        this.wsim = cfgs["wsim"];
        this.wemb = cfgs["wemb"];
        this.wmar = cfgs["wmar"];

    def forward(this, proto, outcls, outcos, label_flatten):
        proto_loss = trnf.relu(proto[1:].matmul(proto[1:].T) - 0.14).mean();
        w = torch.ones_like(torch.ones(outcls.shape[-1])).to(proto.device).float();
        w[-1] = 0.1;
        # change introduced with va5. Masked timestamp does not contribute to loss.
        # Though we cannot say it's unknown(if the image contains one single character) --- perhaps we can?
        clsloss = trnf.cross_entropy(outcls, label_flatten, w,ignore_index=-1);
        if(this.wmar>0):
            margin_loss = this.url.forward(outcls, label_flatten, 0.5)
        else:
            margin_loss=0;

        if(outcos is not None and this.wsim>0):
            cos_loss = this.cosloss(outcos, label_flatten);
            # ace_loss=this.aceloss(outcls,label_flatten)
            loss = cos_loss * this.wsim + clsloss * this.wcls + margin_loss * this.wmar + this.wemb * proto_loss;
            terms = {
                "total": loss.detach().item(),
                "margin": margin_loss.detach().item(),
                "main": clsloss.detach().item(),
                "sim": cos_loss.detach().item(),
                "emb": proto_loss.detach().item(),
            }
        else:
            loss =  clsloss * this.wcls + margin_loss * this.wmar + this.wemb * proto_loss;
            terms = {
                "total": loss.detach().item(),
                # "margin": margin_loss.detach().item(),
                "main": clsloss.detach().item(),
                "emb": proto_loss.detach().item(),
            }
        return loss, terms
class osdanloss_clsctx(nn.Module):
    def __init__(this, cfgs):
        super(osdanloss_clsctx, this).__init__();
        this.setuploss(cfgs);

    def setuploss(this, cfgs):
        this.criterion_CE = nn.CrossEntropyLoss();
        # this.aceloss=
        this.wcls = cfgs["wcls"];
        this.wctx = cfgs["wctx"];
        this.wshr=cfgs["wshr"];
        this.wpemb=cfgs["wpemb"];
        this.wgemb=cfgs["wgemb"];
        this.reduction=cfgs["reduction"];


    def forward(this, proto, outcls,ctxproto,ctxcls, label_flatten):
        proto_loss = trnf.relu(proto[1:].matmul(proto[1:].T) - 0.14).mean();
        w = torch.ones_like(torch.ones(outcls.shape[-1])).to(proto.device).float();
        w[-1] = 0.1;
        clsloss = trnf.cross_entropy(outcls, label_flatten, w,ignore_index=-1);
        loss =  clsloss * this.wcls  + this.wemb * proto_loss;
        terms = {
            "total": loss.detach().item(),
            "main": clsloss.detach().item(),
            "emb": proto_loss.detach().item(),
        }
        return loss, terms
class osdanloss_clsemb(nn.Module):
    def __init__(this, cfgs):
        super(osdanloss_clsemb, this).__init__();
        this.setuploss(cfgs);

    def label_weight(this, shape, label):
        weight = torch.zeros(shape).to(label.device) + 0.1
        weight = torch.scatter_add(weight, 0, label, torch.ones_like(label).float());
        weight = 1. / weight;
        weight[-1] /= 200;
        return weight;

    def setuploss(this, cfgs):
        this.criterion_CE = nn.CrossEntropyLoss();
        # this.aceloss=
        this.wcls = cfgs["wcls"];
        this.wemb = cfgs["wemb"];
        this.reduction=cfgs["reduction"];

    def forward(this, proto, outcls, label_flatten):
        proto_loss = trnf.relu(proto[1:].matmul(proto[1:].T) - 0.14).mean();
        w = torch.ones_like(torch.ones(outcls.shape[-1])).to(proto.device).float();
        w[-1] = 0.1;
        clsloss = trnf.cross_entropy(outcls, label_flatten, w,ignore_index=-1);
        loss =  clsloss * this.wcls  + this.wemb * proto_loss;
        terms = {
            "total": loss.detach().item(),
            "main": clsloss.detach().item(),
            "emb": proto_loss.detach().item(),
        }
        return loss, terms
class fsldanloss_clsembohem(nn.Module):
    def __init__(this, cfgs):
        super(fsldanloss_clsembohem, this).__init__();
        this.setuploss(cfgs);


    def setuploss(this, cfgs):
        this.criterion_CE = nn.CrossEntropyLoss();
        # this.aceloss=
        this.wcls = cfgs["wcls"];
        this.wemb = cfgs["wemb"];
        this.dirty_frac=cfgs["dirty_frac"];
        this.too_simple_frac=cfgs["too_simple_frac"];


    def forward(this, proto, outcls, label_flatten):
        if(this.wemb>0):
            proto_loss = trnf.relu(proto[1:].matmul(proto[1:].T) - 0.14).mean();
        else:
            proto_loss=torch.tensor(0.).float();
        clsloss = trnf.cross_entropy(outcls, label_flatten,reduction="none");
        with torch.no_grad():
            w = torch.ones_like(clsloss,device=proto.device).float();
            # 20% dirty data. The model drops 20% samples that show less agreement to history experience
            # These predictions are expected to be handled by the ctx module following.
            tpk=int(clsloss.shape[0]*this.too_simple_frac);
            if(tpk>0):
                w[torch.topk(clsloss,tpk,0,largest=False)[1]]=0;
                w[clsloss>0.5]=1;
            w[torch.topk(clsloss, int(clsloss.shape[0] * this.dirty_frac), 0, largest=True)[1]] = 0;
            # w[label_flatten==proto.shape[0]]=1;
        clsloss=(w*clsloss).sum()/(w.sum()+0.00001);
        loss =  clsloss * this.wcls  + this.wemb * proto_loss;
        terms = {
            "total": loss.detach().item(),
            "main": clsloss.detach().item(),
            "emb": proto_loss.detach().item(),
        }
        return loss, terms
class osdanloss_trident(nn.Module):
    def __init__(this, cfgs):
        super(osdanloss_trident, this).__init__();
        this.setuploss(cfgs);

    def label_weight(this, shape, label):
        weight = torch.zeros(shape).to(label.device) + 0.1
        weight = torch.scatter_add(weight, 0, label, torch.ones_like(label).float());
        weight = 1. / weight;
        weight[-1] /= 200;
        return weight;

    def setuploss(this, cfgs):
        # this.aceloss=
        this.wcls = cfgs["wcls"];
        this.wemb = cfgs["wemb"];
        this.wrew=cfgs["wrew"];
        this.ppr=cfgs["ppr"];


    def get_scatter_region(this,labels,length):
        target=torch.zeros(length)
        beg=0;
        id=0;
        for l in labels:
            le=len(l)+1;
            target[beg:beg+le]=id;
            id+=1;
            beg+=le;
        return target;
    def compute_rewards(this,choutputs,labels):
        rew=torch.zeros(len(labels),len(choutputs))
        for i in range(len(labels)):
            for j in range(len(choutputs)):
                rew[i][j]=1-(pylcs.edit_distance(choutputs[j][i],labels[i]))/(len(labels[i])+0.0001);
        # minus baseline trick, seems I forgot to detach something(www)
        # TODO: check whether we need to detach the mean. 
        rew[:,1:]-=0.02 # prefer the square option(which does not require encoding the picture again),
                                  # however if the situation requires invoke the necessary branch
        rrew=rew - rew.mean(dim=1, keepdim=True)

        return rrew;


    def forward(this,proto,outclss,outcoss,choutputs,branch_logit,branch_prediction,labels,label_flatten):
        proto_loss = trnf.relu(proto[1:].matmul(proto[1:].T) - 0.14).mean();
        w = torch.ones_like(torch.ones(outclss[0].shape[-1])).to(proto.device).float();
        w[-1] = 0.1;
        sr = this.get_scatter_region(labels, outclss[0].shape[0])
        cls_losses=[];
        rew=this.compute_rewards(choutputs,labels);
        for i in range(len(outclss)):
            clsloss = trnf.cross_entropy(outclss[i], label_flatten, w,reduction="none");
            scl=torch_scatter.scatter_mean(clsloss,sr.long().cuda());
            cls_losses.append(scl);
        # ace_loss=this.aceloss(outcls,label_flatten)
        # Vodoo to keep each branch alive. 1.9=0.3+1
        tw=(1+this.ppr*len(outclss));
        weight=(branch_prediction.detach()+this.ppr)/tw;

        clslossw=(torch.stack(cls_losses,1)*weight).sum(1).mean();

        arew= (rew.cuda()*branch_prediction).sum(1).mean();
        loss =  clslossw * this.wcls + this.wemb * proto_loss-this.wrew*arew;
        terms = {
            "total": loss.detach().item(),
            "main": clslossw.detach().item(),
            "emb": proto_loss.detach().item(),
            "Exp. 1-ned": arew.detach().item(),
        }
        return loss, terms

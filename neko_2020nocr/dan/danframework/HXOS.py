from torch.utils.data import DataLoader
#------------------------
from neko_2020nocr.dan.utils import *
from neko_2020nocr.dan.common.common import display_cfgs,load_dataset,Zero_Grad,Train_or_Eval,generate_optimizer,Updata_Parameters,flatten_label
from neko_2020nocr.dan.common.common_xos import load_network;
from neko_2020nocr.dan.danframework.neko_abstract_dan import neko_abstract_DAN;
from neko_sdk.ocr_modules.trainable_losses.neko_url import neko_unknown_ranking_loss;
from neko_2020nocr.dan.visdan import visdan;
from neko_sdk.ocr_modules.neko_confusion_matrix import neko_confusion_matrix
import os;
from neko_sdk.ocr_modules.trainable_losses.neko_url import neko_unknown_ranking_loss;
# from torch_scatter import scatter_mean;
from neko_sdk.ocr_modules.trainable_losses.cosloss import neko_cos_loss2,neko_cos_loss;

class neko_cos_loss3(nn.Module):
    def __init__(this):
        super(neko_cos_loss3, this).__init__()
        pass;
    def forward(this,pred,gt,weight=None):
        oh=torch.zeros_like(pred).scatter_(1, gt.unsqueeze(-1), 1)[:,1:-1];
        noh=1-oh;
        # oh*=weight[1:-1].unsqueeze(0);
        # noh *= weight[1:-1].unsqueeze(0);
        mask=oh.max(dim=-1,keepdim=True)[0];
        oh*=mask;
        noh*=mask;
        pred_=pred[:,1:-1];
        corr=oh*pred_;
        ### Only classes too close to each other should be pushed.
        ### That said if they are far we don't need to align them
        ### Again 0.14 =cos(spatial angluar on 50k evenly distributed prototype)
        wrong=torch.nn.functional.relu(noh*pred_-0.14)
        hwrong=torch.max(torch.nn.functional.relu(noh*pred_-0.14),dim=1)[0]

        nl=torch.sum(wrong)/(torch.sum(noh)+0.009)+hwrong.mean();
        pl=1-torch.sum(corr)/(torch.sum(oh)+0.009)
        return (nl+pl)/2;
# HSOS, HDOS, HDOSCS
# the main feature is that the DPE returns embeddings for characters.
class HXOS(neko_abstract_DAN):
    def setuploss(this):
        this.criterion_CE = nn.CrossEntropyLoss().cuda()
    def load_network(this):
        return load_network(this.cfgs);
    def get_ar_cntr(this,key,case_sensitive):
        return neko_os_Attention_AR_counter(key,case_sensitive);
    def get_loss_cntr(this,show_interval):
        return Loss_counter(show_interval);


    def test(this,test_loader, model, tools,miter=1000,debug=False,dbgpath=None):
        Train_or_Eval(model, 'Eval')
        proto,semb, plabel, tdict = model[3].dump_all();
        i=0;
        visualizer=None;
        if dbgpath is not None:
            visualizer=visdan(dbgpath);
        cfm=neko_confusion_matrix();

        for sample_batched in test_loader:
            if i>miter:
                break;
            i+=1;
            data = sample_batched['image']
            label = sample_batched['label'];
            target = model[3].encode(proto, plabel, tdict, label)

            data = data.cuda()
            target = target
            label_flatten, length = tools[1](target)
            target, label_flatten = target.cuda(), label_flatten.cuda()

            features = model[0](data)
            A = model[1](features)
            # A0=A.detach().clone();
            output, out_length,A = model[2](features[-1], proto,semb, plabel, A, None, length, True)
            # A=A.max(dim=2)[0];
            choutput, prdt_prob= model[3].decode(output, out_length, proto, plabel, tdict);
            tools[0].add_iter(choutput, out_length, label,debug)

            for i in range(len(choutput)):
                cfm.addpairquickandugly(choutput[i],label[i]);

            if(visualizer is not None):
                visualizer.addbatch(data, A, label,choutput)
        if(dbgpath):
            try:
                cfm.save_matrix(os.path.join())
            except:
                pass;
        tools[0].show();
        Train_or_Eval(model, 'Train')
    def runtest(this,miter=1000,debug=False,dpgpath=None):
        this.test((this.test_loader), this.model, [this.test_acc_counter,
                                                   flatten_label,
                                                   ],miter=miter,debug=debug,dbgpath=dpgpath);
        this.test_acc_counter.clear();

    def mk_proto(this,label):
        return None,None,None,None;


    def fpbp(this, data, label,cased=None):
        proto,semb, plabel, tdict = this.mk_proto(label);
        target = this.model[3].encode(proto, plabel, tdict, label);

        Train_or_Eval(this.model, 'Train')
        data = data.cuda()
        label_flatten, length = flatten_label(target)
        target, label_flatten = target.cuda(), label_flatten.cuda()
        # net forward
        features = this.model[0](data)
        A = this.model[1](features)
        # feature,protos,labels, A, hype, text_length, test = False)
        output, _ = this.model[2](features[-1], proto,semb, plabel, A, target, length)
        choutput, prdt_prob, = this.model[3].decode(output, length, proto, plabel, tdict);
        # computing accuracy and loss
        tarswunk = ["".join([tdict[i.item()] for i in target[j]]).replace('[s]', "") for j in
                    range(len(target))];
        this.train_acc_counter.add_iter(choutput, length, tarswunk)
        loss = this.criterion_CE(output, label_flatten);
        this.loss_counter.add_iter(loss)
        # update network
        Zero_Grad(this.model)
        loss.backward()

class HSOS(HXOS):
    def mk_proto(this,label):
        return this.model[3].dump_all()

class HDOS(HXOS):
    def mk_proto(this,label):
        return this.model[3].sample_tr(label)

class HXOSC(HXOS):
    def setuploss(this):
        this.criterion_CE = nn.CrossEntropyLoss().cuda()
        this.cosloss=neko_cos_loss().cuda();

    def fpbp(this, data, label,cased=None):
        proto, semb, plabel, tdict = this.mk_proto(label);
        target = this.model[3].encode(proto, plabel, tdict, label);

        Train_or_Eval(this.model, 'Train')
        data = data.cuda()
        label_flatten, length = flatten_label(target)
        target, label_flatten = target.cuda(), label_flatten.cuda()
        # net forward
        features = this.model[0](data)
        A = this.model[1](features)
        # feature,protos,labels, A, hype, text_length, test = False)
        outcls, outcos = this.model[2](features[-1], proto, semb, plabel, A, target, length)
        choutput, prdt_prob, = this.model[3].decode(outcls, length, proto, plabel, tdict);
        # computing accuracy and loss
        tarswunk = ["".join([tdict[i.item()] for i in target[j]]).replace('[s]', "") for j in
                    range(len(target))];
        this.train_acc_counter.add_iter(choutput, length, tarswunk)
        clsloss = this.criterion_CE(outcls, label_flatten);
        cos_loss= this.cosloss(outcos,label_flatten);
        loss=cos_loss+clsloss;
        this.loss_counter.add_iter(loss)
        # update network
        Zero_Grad(this.model)
        loss.backward()

class HDOSC(HXOSC):
    def mk_proto(this,label):
        return this.model[3].sample_tr(label)




class HXOSCR(HXOSC):
    def setuploss(this):
        this.criterion_CE = nn.CrossEntropyLoss().cuda()
        this.url=neko_unknown_ranking_loss();
        this.cosloss=neko_cos_loss().cuda();
        this.wcls = 1;
        this.wsim = 1;
        this.wmar = 0;
    def fpbp(this, data, label,cased=None):
        proto, semb, plabel, tdict = this.mk_proto(label);
        target = this.model[3].encode(proto, plabel, tdict, label);

        Train_or_Eval(this.model, 'Train')
        data = data.cuda()
        label_flatten, length = flatten_label(target)
        target, label_flatten = target.cuda(), label_flatten.cuda()
        # net forward
        features = this.model[0](data)
        A = this.model[1](features)
        # feature,protos,labels, A, hype, text_length, test = False)
        outcls, outcos = this.model[2](features[-1], proto,semb , plabel, A, target, length)
        choutput, prdt_prob, = this.model[3].decode(outcls, length, proto, plabel, tdict);

        # computing accuracy and loss
        tarswunk = ["".join([tdict[i.item()] for i in target[j]]).replace('[s]', "") for j in
                    range(len(target))];
        this.train_acc_counter.add_iter(choutput, length, tarswunk)
        clsloss = this.criterion_CE(outcls, label_flatten);
        cos_loss= this.cosloss(outcos,label_flatten);
        margin_loss = this.url.forward(outcls, label_flatten, 0.5)
        loss=cos_loss*this.wsim+clsloss*this.wcls+margin_loss*this.wmar;
        terms={
            "total": loss.detach().item(),
            "margin": margin_loss.detach().item(),
            "main": clsloss.detach().item(),
            "sim":cos_loss.detach().item(),
        }
        this.loss_counter.add_iter(loss,terms)
        # update network
        Zero_Grad(this.model)
        loss.backward()
class HDOSCR(HXOSCR):
    def mk_proto(this,label):
        return this.model[3].sample_tr(label)




class HXOSCRR(HXOSC):
    def setuploss(this):
        this.criterion_CE = nn.CrossEntropyLoss().cuda()
        this.url=neko_unknown_ranking_loss();
        this.cosloss=neko_cos_loss().cuda();
        this.wcls = 1;
        this.wsim = 1;
        this.wmar = 0.3;
    def fpbp(this, data, label,cased=None):
        proto, semb, plabel, tdict = this.mk_proto(label);
        target = this.model[3].encode(proto, plabel, tdict, label);

        Train_or_Eval(this.model, 'Train')
        data = data.cuda()
        label_flatten, length = flatten_label(target)
        target, label_flatten = target.cuda(), label_flatten.cuda()
        # net forward
        features = this.model[0](data)
        A = this.model[1](features)
        # feature,protos,labels, A, hype, text_length, test = False)
        outcls, outcos = this.model[2](features[-1], proto,semb , plabel, A, target, length)
        choutput, prdt_prob, = this.model[3].decode(outcls, length, proto, plabel, tdict);

        # computing accuracy and loss
        tarswunk = ["".join([tdict[i.item()] for i in target[j]]).replace('[s]', "") for j in
                    range(len(target))];
        this.train_acc_counter.add_iter(choutput, length, tarswunk)
        clsloss = this.criterion_CE(outcls, label_flatten);
        cos_loss= this.cosloss(outcos,label_flatten);
        margin_loss = this.url.forward(outcls, label_flatten, 0.5)
        loss=cos_loss*this.wsim+clsloss*this.wcls+margin_loss*this.wmar;
        terms={
            "total": loss.detach().item(),
            "margin": margin_loss.detach().item(),
            "main": clsloss.detach().item(),
            "sim":cos_loss.detach().item(),
        }
        this.loss_counter.add_iter(loss,terms)
        # update network
        Zero_Grad(this.model)
        loss.backward()
class HDOSCRR(HXOSCRR):
    def mk_proto(this,label):
        return this.model[3].sample_tr(label)
class HXOSCO(HXOSC):
    def setuploss(this):
        this.criterion_CE = nn.CrossEntropyLoss().cuda()
        this.url=neko_unknown_ranking_loss();
        this.cosloss=neko_cos_loss().cuda();
        this.wcls = 0;
        this.wsim = 1;
        this.wmar = 0;
    def fpbp(this, data, label,cased=None):
        proto, semb, plabel, tdict = this.mk_proto(label);
        target = this.model[3].encode(proto, plabel, tdict, label);

        Train_or_Eval(this.model, 'Train')
        data = data.cuda()
        label_flatten, length = flatten_label(target)
        target, label_flatten = target.cuda(), label_flatten.cuda()
        # net forward
        features = this.model[0](data)
        A = this.model[1](features)
        # feature,protos,labels, A, hype, text_length, test = False)
        outcls, outcos = this.model[2](features[-1], proto,semb , plabel, A, target, length)
        choutput, prdt_prob, = this.model[3].decode(outcls, length, proto, plabel, tdict);

        # computing accuracy and loss
        tarswunk = ["".join([tdict[i.item()] for i in target[j]]).replace('[s]', "") for j in
                    range(len(target))];
        this.train_acc_counter.add_iter(choutput, length, tarswunk)
        clsloss = this.criterion_CE(outcls, label_flatten);
        cos_loss= this.cosloss(outcos,label_flatten);
        margin_loss = this.url.forward(outcls, label_flatten, 0.5)
        loss=cos_loss*this.wsim+clsloss*this.wcls+margin_loss*this.wmar;
        terms={
            "total": loss.detach().item(),
            "margin": margin_loss.detach().item(),
            "main": clsloss.detach().item(),
            "sim":cos_loss.detach().item(),
        }
        this.loss_counter.add_iter(loss,terms)
        # update network
        Zero_Grad(this.model)
        loss.backward()
class HDOSCO(HXOSCRR):
    def mk_proto(this,label):
        return this.model[3].sample_tr(label)

class HXOSCB(HXOSC):
    def setuploss(this):
        this.criterion_CE = nn.CrossEntropyLoss().cuda()
        this.url=neko_unknown_ranking_loss();
        this.cosloss=neko_cos_loss().cuda();
        this.wcls = 1;
        this.wsim = 0;
        this.wmar = 0;
    def fpbp(this, data, label,cased=None):
        proto, semb, plabel, tdict = this.mk_proto(label);
        target = this.model[3].encode(proto, plabel, tdict, label);

        Train_or_Eval(this.model, 'Train')
        data = data.cuda()
        label_flatten, length = flatten_label(target)
        target, label_flatten = target.cuda(), label_flatten.cuda()
        # net forward
        features = this.model[0](data)
        A = this.model[1](features)
        # feature,protos,labels, A, hype, text_length, test = False)
        outcls, outcos = this.model[2](features[-1], proto,semb , plabel, A, target, length)
        choutput, prdt_prob, = this.model[3].decode(outcls, length, proto, plabel, tdict);

        # computing accuracy and loss
        tarswunk = ["".join([tdict[i.item()] for i in target[j]]).replace('[s]', "") for j in
                    range(len(target))];
        this.train_acc_counter.add_iter(choutput, length, tarswunk)
        clsloss = this.criterion_CE(outcls, label_flatten);
        cos_loss= this.cosloss(outcos,label_flatten);
        margin_loss = this.url.forward(outcls, label_flatten, 0.5)
        loss=cos_loss*this.wsim+clsloss*this.wcls+margin_loss*this.wmar;
        terms={
            "total": loss.detach().item(),
            "margin": margin_loss.detach().item(),
            "main": clsloss.detach().item(),
            "sim":cos_loss.detach().item(),
        }
        this.loss_counter.add_iter(loss,terms)
        # update network
        Zero_Grad(this.model)
        loss.backward()
class HDOSB(HXOSCB):
    def mk_proto(this,label):
        return this.model[3].sample_tr(label)

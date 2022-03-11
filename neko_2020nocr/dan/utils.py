import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2
import json
import editdistance as ed
import regex
class cha_encdec():
    def __init__(self, dict_file, case_sensitive = True):
        self.dict = []
        self.case_sensitive = case_sensitive
        lines = open(dict_file , 'r').readlines()
        for line in lines:
            self.dict.append(line.replace('\n', ''))
    def encode(self, label_batch):
        max_len = max([len(regex.findall(r'\X', s, regex.U) ) for s in label_batch])
        out = torch.zeros(len(label_batch), max_len+1).long()
        for i in range(0, len(label_batch)):
            if not self.case_sensitive:
                cur_encoded = torch.tensor([self.dict.index(char.lower()) if char.lower() in self.dict else len(self.dict)
                                     for char in regex.findall(r'\X', label_batch[i], regex.U) ]) + 1
            else:
                cur_encoded = torch.tensor([self.dict.index(char) if char in self.dict else len(self.dict)
                                     for char in regex.findall(r'\X', label_batch[i], regex.U) ]) + 1
            out[i][0:len(cur_encoded)] = cur_encoded
        return out
    def decode(self, net_out, length):
    # decoding prediction into text with geometric-mean probability
    # the probability is used to select the more realiable prediction when using bi-directional decoders
        out = []
        out_prob = [] 
        net_out = F.softmax(net_out, dim = 1)
        for i in range(0, length.shape[0]):
            current_idx_list = net_out[int(length[:i].sum()) : int(length[:i].sum() + length[i])].topk(1)[1][:,0].tolist()
            current_text = ''.join([self.dict[_-1] if _ > 0 and _ <= len(self.dict) else '⑨' for _ in current_idx_list])
            current_probability = net_out[int(length[:i].sum()) : int(length[:i].sum() + length[i])].topk(1)[0][:,0]
            current_probability = torch.exp(torch.log(current_probability).sum() / current_probability.size()[0])
            out.append(current_text)
            out_prob.append(current_probability)
        return (out, out_prob)

class neko_osfsl_ACR_counter():
    def __init__(self, display_string):
        self.correct = 0
        self.total_samples = 0.
        self.display_string = display_string
    def clear(self):
        self.correct = 0
        self.total_samples = 0.

    def add_iter(self,pred, labels):
        cnt=len(labels);
        self.total_samples +=cnt;
        for i in range(cnt):
            if(pred[i]==labels[i]):
                self.correct+=1;

    def show(self):
        # Accuracy for scene text.
        # CER and WER for handwritten text.
        print(self.display_string)
        if self.total_samples == 0:
            pass
        print('Accuracy: {:.6f}'.format(
            self.correct / self.total_samples,))

class neko_os_ACR_counter():
    def __init__(self, display_string):
        self.correct = 0
        self.total_samples = 0.
        self.display_string = display_string
    def clear(self):
        self.correct = 0
        self.total_samples = 0.

    def add_iter(self,pred, labels):
        self.total_samples += labels.shape[0]
        self.correct+=(labels==pred).sum().item();

    def show(self):
        # Accuracy for scene text.
        # CER and WER for handwritten text.
        print(self.display_string)
        if self.total_samples == 0:
            pass
        print('Accuracy: {:.6f}'.format(
            self.correct / self.total_samples,))

class neko_os_Attention_AR_counter():
    def __init__(self, display_string, case_sensitive):
        self.correct = 0
        self.total_samples = 0.
        self.distance_C = 0
        self.total_C = 0.
        self.distance_W = 0
        self.total_W = 0.
        self.display_string = display_string
        self.case_sensitive = case_sensitive

    def clear(self):
        self.correct = 0
        self.total_samples = 0.
        self.distance_C = 0
        self.total_C = 0.
        self.distance_W = 0
        self.total_W = 0.

    def add_iter(self,prdt_texts, label_length, labels,debug=False):
        if(labels is None):
            return ;
        start = 0
        start_o = 0
        if debug:
            for i in range(len(prdt_texts)):
                print(labels[i], "->-", prdt_texts[i]);

        self.total_samples += len(labels)
        for i in range(0, len(prdt_texts)):
            if not self.case_sensitive:
                prdt_texts[i] = prdt_texts[i].lower().replace("⑨","")
                labels[i] = labels[i].lower()
            all_words = []
            for w in labels[i].split('|sadhkjashfkjasyhf') + prdt_texts[i].split('||sadhkjashfkjasyhf'):
                if w not in all_words:
                    all_words.append(w)
            l_words = [all_words.index(_) for _ in labels[i].split('||sadhkjashfkjasyhf')]
            p_words = [all_words.index(_) for _ in prdt_texts[i].split('||sadhkjashfkjasyhf')]
            self.distance_C += ed.eval(labels[i], prdt_texts[i])
            self.distance_W += ed.eval(l_words, p_words)
            self.total_C += len(labels[i])
            self.total_W += len(l_words)
            self.correct = self.correct + 1 if labels[i] == prdt_texts[i] else self.correct

    def show(self):
        # Accuracy for scene text.
        # CER and WER for handwritten text.
        print(self.display_string)
        if self.total_samples == 0:
            pass
        print('Accuracy: {:.6f}, AR: {:.6f}, CER: {:.6f}, WER: {:.6f}'.format(
            self.correct / self.total_samples,
            1 - self.distance_C / self.total_C,
            self.distance_C / self.total_C,
            self.distance_W / self.total_W))

class Attention_AR_counter():
    def __init__(self, display_string, dict_file, case_sensitive):
        self.correct = 0
        self.total_samples = 0.
        self.distance_C = 0
        self.total_C = 0.
        self.distance_W = 0
        self.total_W = 0.
        self.display_string = display_string
        self.case_sensitive = case_sensitive
        self.de = cha_encdec(dict_file, case_sensitive)

    def clear(self):
        self.correct = 0
        self.total_samples = 0.
        self.distance_C = 0
        self.total_C = 0.
        self.distance_W = 0
        self.total_W = 0.
        
    def add_iter(self, output, out_length, label_length, labels,debug=False):
        self.total_samples += label_length.size()[0]
        raw_prdts = output.topk(1)[1]
        prdt_texts, prdt_prob = self.de.decode(output, out_length)
        CS=[]
        DS=[];
        batch_corr=0;
        batch_tot=0;
        if debug:
            for i in range(len(prdt_texts)):
                print(labels[i].lower(),"->-",prdt_texts[i].lower());
        prdt_texts=[i.replace("⑨","") for i in prdt_texts];
        for i in range(0, len(prdt_texts)):
            if not self.case_sensitive:
                prdt_texts[i] = prdt_texts[i].lower()
                labels[i] = labels[i].lower()
            all_words = []
            for w in labels[i].split('|') + prdt_texts[i].split('|'):
                if w not in all_words:
                    all_words.append(w)
            l_words = [all_words.index(_) for _ in labels[i].split('|')]
            p_words = [all_words.index(_) for _ in prdt_texts[i].split('|')]
            self.distance_C += ed.eval(labels[i], prdt_texts[i])
            self.distance_W += ed.eval(l_words, p_words)
            self.total_C += len(labels[i])
            self.total_W += len(l_words)
            CS.append(len(labels[i]));
            DS.append(ed.eval(labels[i], prdt_texts[i]));
            self.correct = self.correct + 1 if labels[i] == prdt_texts[i] else self.correct
            batch_corr = batch_corr + 1 if labels[i] == prdt_texts[i] else batch_corr;
            batch_tot+=1;
        return batch_corr/batch_tot,CS,DS;
    def show(self):
    # Accuracy for scene text. 
    # CER and WER for handwritten text.
        print(self.display_string)
        if self.total_samples == 0:
            pass
        print('Accuracy: {:.6f}, AR: {:.6f}, CER: {:.6f}, WER: {:.6f}'.format(
            self.correct / self.total_samples,
            1 - self.distance_C / self.total_C,
            self.distance_C / self.total_C,
            self.distance_W / self.total_W))


class Attention_AR_counter_node():
    def __init__(self, display_string, case_sensitive):
        self.correct = 0
        self.total_samples = 0.
        self.distance_C = 0
        self.total_C = 0.
        self.distance_W = 0
        self.total_W = 0.
        self.display_string = display_string
        self.case_sensitive = case_sensitive

    def clear(self):
        self.correct = 0
        self.total_samples = 0.
        self.distance_C = 0
        self.total_C = 0.
        self.distance_W = 0
        self.total_W = 0.

    def add_iter(self, output, out_length, label_length, labels,de, debug=False):
        self.total_samples += label_length.size()[0]
        raw_prdts = output.topk(1)[1]
        prdt_texts, prdt_prob = de.decode(output, out_length)
        if debug:
            for i in range(len(prdt_texts)):
                print(labels[i].lower(), "->-", prdt_texts[i].lower());
        prdt_texts = [i.replace("⑨", "") for i in prdt_texts];
        for i in range(0, len(prdt_texts)):
            if not self.case_sensitive:
                prdt_texts[i] = prdt_texts[i].lower()
                labels[i] = labels[i].lower()
            all_words = []
            for w in labels[i].split('|') + prdt_texts[i].split('|'):
                if w not in all_words:
                    all_words.append(w)
            l_words = [all_words.index(_) for _ in labels[i].split('|')]
            p_words = [all_words.index(_) for _ in prdt_texts[i].split('|')]
            self.distance_C += ed.eval(labels[i], prdt_texts[i])
            self.distance_W += ed.eval(l_words, p_words)
            self.total_C += len(labels[i])
            self.total_W += len(l_words)
            self.correct = self.correct + 1 if labels[i] == prdt_texts[i] else self.correct

    def show(self):
        # Accuracy for scene text.
        # CER and WER for handwritten text.
        print(self.display_string)
        if self.total_samples == 0:
            pass
        print('Accuracy: {:.6f}, AR: {:.6f}, CER: {:.6f}, WER: {:.6f}'.format(
            self.correct / self.total_samples,
            1 - self.distance_C / self.total_C,
            self.distance_C / self.total_C,
            self.distance_W / self.total_W))

class Loss_counter():
    def __init__(self, display_interval):
        self.display_interval = display_interval
        self.total_iters = 0.
        self.loss_sum = 0
        self.termsum={};
    def mkterm(self,dic,prfx=""):
        for k in dic:
            if(type(dic[k])==dict):
                self.mkterm(dic,prfx+k)
            if k not in self.termsum:
                self.termsum[prfx+k] = 0;
            self.termsum[prfx+k] += float(dic[k])

    def add_iter(self, loss,terms=None):
        self.total_iters += 1
        self.loss_sum += float(loss)
        if terms is not None:
            self.mkterm(terms);

    def clear(self):
        self.total_iters = 0
        self.loss_sum = 0
        self.termsum={};
    def show(self):
        print( self.get_loss_and_terms());
    def get_loss(self):
        loss = self.loss_sum / self.total_iters if self.total_iters > 0 else 0
        self.clear();
        return loss
    def get_loss_and_terms(this):
        loss = this.loss_sum / this.total_iters if this.total_iters > 0 else 0
        retterms={};
        for k in this.termsum:
            term = this.termsum[k] / this.total_iters if this.total_iters > 0 else 0;
            retterms[k]=term;
        this.clear();
        return loss,retterms;


class neko_oswr_Attention_AR_counter():
    def __init__(this, display_string, case_sensitive):
        this.clear()
        this.display_string = display_string
        this.case_sensitive = case_sensitive

    def clear(this):
        this.correct = 0
        this.total_samples = 0.
        this.total_C = 0.
        this.total_W = 0.
        this.total_U=0.
        this.total_K=0.
        this.Ucorr=0.
        this.Kcorr=0.
        this.KtU=0.

    def add_iter(this,prdt_texts, label_length, labels,debug=False):
        if(labels is None):
            return ;
        start = 0
        start_o = 0
        if debug:
            for i in range(len(prdt_texts)):
                print(labels[i], "->-", prdt_texts[i]);

        this.total_samples += label_length.size()[0]
        for i in range(0, len(prdt_texts)):
            if not this.case_sensitive:
                prdt_texts[i] = prdt_texts[i].lower()
                labels[i] = labels[i].lower()
            all_words = []
            for w in labels[i].split('|sadhkjashfkjasyhf') + prdt_texts[i].split('||sadhkjashfkjasyhf'):
                if w not in all_words:
                    all_words.append(w)
            l_words = [all_words.index(_) for _ in labels[i].split('||sadhkjashfkjasyhf')]
            p_words = [all_words.index(_) for _ in prdt_texts[i].split('||sadhkjashfkjasyhf')]
            this.total_C += len(labels[i])
            this.total_W += len(l_words)
            cflag=int(labels[i] == prdt_texts[i]);
            this.correct = this.correct + cflag;
            if(labels[i].find("⑨")!=-1):
                this.total_U+=1;
                this.Ucorr+=(prdt_texts[i].find("⑨")!=-1);
            else:
                this.total_K+=1;
                this.Kcorr+=cflag;
                this.KtU+=(prdt_texts[i].find("⑨")!=-1);

    def show(this):
        # Accuracy for scene text.
        # CER and WER for handwritten text.
        print(this.display_string)
        if this.total_samples == 0:
            pass
        R=this.Ucorr / this.total_U;
        P=this.Ucorr / (this.Ucorr + this.KtU);
        F=2*(R*P)/(R+P)
        print('Accuracy: {:.6f}, KACR: {:.6f},URCL:{:.6f}, UPRE {:.6f}, F {:.6f}'.format(
            this.correct / this.total_samples,
            this.Kcorr / this.total_K,
            R,P,F

        ))
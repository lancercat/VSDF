from torch import nn
import torch
class neko_pystat(nn.Module):
    def __init__(this,max_capacity=900000):
        super(neko_pystat, this).__init__();
        this.cnts=torch.nn.Parameter(torch.zeros(max_capacity),requires_grad=False);
        this.total = torch.nn.Parameter(torch.tensor(1e-9), requires_grad=False);
        this.lclipping_freq=0.01;
        this.cdict={};
        
    def _save_to_state_dict(this, destination, prefix, keep_vars):
        destination[prefix + "cdict"]=this.cdict
        super(neko_pystat, this)._save_to_state_dict(destination,prefix,keep_vars)
    def _load_from_state_dict(this, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        this.cdict=state_dict[prefix+"cdict"];
        del state_dict[prefix+"cdict"]
        super(neko_pystat, this)._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs);
    def forward_train(this,flatten_label,gdict,llen):
        for i in range(flatten_label.shape[0]):
            ch=gdict[flatten_label[i].item()];
            if(ch not in this.cdict):
                this.cdict[ch]=len(this.cdict);
            this.cnts[this.cdict[ch]]+=1;
            this.total+=1;
        return this.forward_eval(gdict,llen)
    def forward_eval(this,gdict,llen):
        ret=torch.zeros(llen,dtype=torch.float,device=this.cnts.device);
        for i in range(llen):
            ch=gdict[i];
            if(ch not in this.cdict):
                this.cdict[ch]=len(this.cdict);
            ret[i]=this.cnts[this.cdict[ch]];
        return torch.clip(ret/this.total,this.lclipping_freq);
    def forward(this,gdict,flatten_label,llen):
        # floats rounds at 16,777,216, we assume the estimation is good enough when it saw ~16M characters.
        if(this.training and this.total<16777009):
            return this.forward_train(flatten_label,gdict,llen);
        else:
            return this.forward_eval(gdict,llen);
if __name__ == '__main__':
    a=neko_pystat(9);
    a.cdict["a"]=9;
    torch.save(a.state_dict(), "test.pt");
    b=neko_pystat(9);
    b.load_state_dict(torch.load("test.pt"));
    print(b.cdict["a"])

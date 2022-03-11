from torch.utils.data import DataLoader
import time;
import torch;

class neko_joint_loader:
    def __init__(this,dataloadercfgs,length):
        this.dataloaders=[];
        this.ddict={};
        this.names=[];
        i=0;
        setcfgs=dataloadercfgs["subsets"]
        for name in setcfgs:
            this.ddict[name]=i;
            i+=1;
            this.names.append(name)
            cfg=setcfgs[name];
            train_data_set = cfg['type'](**cfg['ds_args'])
            train_loader = DataLoader(train_data_set, **cfg['dl_args'])
            this.dataloaders.append(train_loader);
        this.iters=[iter(loader) for loader in this.dataloaders]
        this.length=length;
    def next(this):
        ret={}
        for name in this.names:
            id=this.ddict[name];
            try:
                # use some nonsense that always trigger the reset loader behaviour
                # nep[1000]=1;
                rett=this.iters[id].__next__()
            except:
                a=this.iters[id];
                this.iters[id]=None;
                del a;
                time.sleep(2)  # Prevent possible deadlock during epoch transition
                this.iters[id]=iter(this.dataloaders[id]);
                rett = this.iters[id].__next__()

            for t in rett:
                if (torch.is_tensor(rett[t])):
                    ret[name + "_" + t] = rett[t].contiguous();
                else:
                    ret[name + "_" + t] = rett[t];
        return ret;

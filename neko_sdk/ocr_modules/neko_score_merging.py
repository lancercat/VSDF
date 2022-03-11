from torch_scatter import scatter_max,scatter_mean;
def id_cvt(pred,label):
    return pred;
def scatter_cvt_d(pred,label,dim=-1):
    dev=pred.device;
    label=label.long().to(dev);
    pred=pred.cpu();
    label=label.cpu();
    return scatter_max(pred,label,dim)[0].cuda();

def scatter_cvt(pred, label, dim=-1):
    # The old one seems causing locking problem when launched in parallel. Will see after updating to driver 465
    # return scatter_cvt_d(pred,label,dim)
    dev = pred.device;
    label = label.long().to(dev);
    return scatter_max(pred,label,dim)[0];

def scatter_cvt2(pred, label, dim=-1):
    # return scatter_cvt_d(pred,label,dim)
    dev = pred.device;
    return scatter_max(pred,label,dim)[0];
if __name__ == '__main__':
    import torch
    # def random_data

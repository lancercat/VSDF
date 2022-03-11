import torch.nn
from torch import nn
from torch.nn import functional as trnf

class spatial_attention(nn.Module):
    def __init__(this,ifc):
        super(spatial_attention, this).__init__()
        this.core=torch.nn.Sequential(
            torch.nn.Conv2d(
                ifc,ifc,(3,3),(1,1),(1,1),
            ),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(ifc),
            torch.nn.Conv2d(ifc,1,(1,1)),
            torch.nn.Sigmoid(),
        );

    def forward(this, input):
        x = input[-1]
        return this.core(x);

class spatial_attention_mk2(spatial_attention):

    def forward(this, input):
        x = input[-2];
        d=input[-1];
        if(x.shape[-1]!=d.shape[-1]):
            x=trnf.interpolate(x,[d.shape[-2],d.shape[-1]]);
        return this.core(x);

class spatial_attention_mk3(spatial_attention):
    def forward(this, input):
        x = input[0];
        d=input[-1];
        if(x.shape[-1]!=d.shape[-1]):
            x=trnf.interpolate(x,[d.shape[-2],d.shape[-1]],mode="area");
        return this.core(x);

class spatial_attention_mk4(spatial_attention):

    def forward(this, input):
        x = input[0];
        d = input[-1];
        y = this.core(x);
        if (y.shape[-1] != d.shape[-1]):
            y = trnf.interpolate(y, [d.shape[-2], d.shape[-1]], mode="bilinear");
        return y;
class spatial_attention_mk5(spatial_attention):
    def __init__(this,ifc):
        super(spatial_attention, this).__init__()
        this.core=torch.nn.Sequential(
            torch.nn.Conv2d(
                ifc,ifc,(3,3),(1,1),(1,1),
            ),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(ifc),
            torch.nn.Conv2d(ifc,1,(1,1),bias=False),
            torch.nn.Sigmoid(),
        );

    def forward(this, input):
        x = input[0];
        d = input[-1];
        y = this.core(x);
        if (y.shape[-1] != d.shape[-1]):
            y = trnf.interpolate(y, [d.shape[-2], d.shape[-1]], mode="bilinear");
        return y;

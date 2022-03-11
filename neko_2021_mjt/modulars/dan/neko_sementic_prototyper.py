import torch
from torch.nn import functional as trnf

# note that this version is not dict driven---
# Since torch.module does not provide non-numeric parameter support,
# we offload the labelstr-id mapping to sampler
# This class allows you to get a subset via the sid given by the label sampler, if applicable.
# However, FSL datasets may sample label in dataloaders, which means the sid is needed to be provided by the loader, if necessary.

class neko_sampled_sementic_branch(torch.nn.Module):
    def __init__(this,feat_ch,capacity,spks):
        super(neko_sampled_sementic_branch, this).__init__();
        this.weights=torch.nn.Parameter(torch.rand([capacity,feat_ch])*2-1);
    def sample(this,sids):
        ret = [];
        for i in sids:
            ret.append(this.weights[i]);
        return trnf.normalize(torch.stack(ret), dim=-1)

    def forward(this, sids=None):
        if (sids is None):
            return trnf.normalize(this.weights, dim=-1);
        else:
            return this.sample(sids);

#
# class neko_global_sementic_branch(neko_sampled_sementic_branch):
#     def forward(this,):
#     gids=[]
#         for i in plabel:
#             gids.append(gtdict[tdict[i]]);
#         return this.sample(gids);

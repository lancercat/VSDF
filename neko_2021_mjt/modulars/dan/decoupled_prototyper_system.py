import torch
from torch import nn

import regex

# PROTOENGINE = neko_visual_only_interprinter;


class neko_prototyper_sp(nn.Module):
    def __init__(this,output_channel,spks,has_sem=False):
        super(neko_prototyper_sp,this).__init__()
        this.output_channel=output_channel;
        this.sp_cnt=len(spks);
        this.EOS=0
        this.sp_protos_vis = torch.nn.Parameter(torch.rand([
            this.sp_cnt, this.output_channel]).float() * 2 - 1);
        this.register_parameter("sp_protos_vis", this.sp_protos_vis);
        if(has_sem):
            this.sp_protos_sem = torch.nn.Parameter(torch.rand([
                this.sp_cnt, this.output_channel]).float() * 2 - 1);
            this.register_parameter("sp_protos_sem", this.sp_protos_sem);
        else:
            this.sp_protos_sem=None;

    def forward(this):
        return this.sp_protos_vis,this.sp_protos_sem;

# # We do NOT reduce visual feature in this module. In fact,
# # a cam shall follow the backbone to reduce it thru w and h.
# # the rotation has to be done somewhere else.
# class neko_prototyperpost(nn.Module):
#     def __init__(this, output_channel, spks, dropout=None, capacity=512):
#         super(neko_prototyperpost, this).__init__()
#         this.output_channel = output_channel;
#         this.sp_cnt = len(spks);
#         this.proto_engine = this.PROTOENGINE(this.output_channel);
#         this.dev_ind = torch.nn.Parameter(torch.rand([1]));
#         this.EOS = 0
#         this.sp_protos = torch.nn.Parameter(torch.rand([
#             this.sp_cnt, this.output_channel]).float() * 2 - 1);
#         this.register_parameter("sp_proto", this.sp_protos);
#         if (dropout is not None):
#             this.drop = torch.nn.Dropout(p=0.3);
#         else:
#             this.drop = None;
#         print("DEBUG-SDFGASDFGSDGASFGSD", dropout);
#         # split if too many;
#         this.capacity = capacity;
#         this.freeze_bn_affine = False;
#
#     def freezebn(this):
#         for m in this.modules():
#             if isinstance(m, nn.BatchNorm2d):
#                 m.eval()
#                 if this.freeze_bn_affine:
#                     m.weight.requires_grad = False
#                     m.bias.requires_grad = False
#
#     def unfreezebn(this):
#         for m in this.modules():
#             if isinstance(m, nn.BatchNorm2d):
#                 m.train();
#                 if this.freeze_bn_affine:
#                     m.weight.requires_grad = True
#                     m.bias.requires_grad = True
#
#     def forward(this, vis_proto_srcs,sem_proto_srcs):
#         allproto = trnf.normalize(torch.cat(vis_proto_srcs), dim=1, eps=0.0009);
#         if (this.drop):
#             allproto = this.drop(allproto);
#         pass;
#         return allproto.contiguous();
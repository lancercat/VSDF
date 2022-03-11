import torch
def length_to_mask( length, maxT):
    return (torch.arange(maxT).to(length.device)[None, :] < length[:, None])
import torch
import torch.nn as nn
# standard backbones(Minimum adaption from torchvision)

class Feature_Extractor_hi(nn.Module):
    def __init__(self, strides, compress_layer, input_shape,oupch=512):
        super(Feature_Extractor_hi, self).__init__()
        self.model = resnet.resnet45hi(strides, compress_layer,oupch=oupch,inpch=input_shape[0])
        self.input_shape = input_shape

    def freezebn(this):
        this.model.freezebn();

    def unfreezebn(this):
        this.model.unfreezebn();

    def forward(self, input):
        features = self.model(input.contiguous())
        return features

    def Iwantshapes(self):
        pseudo_input = torch.rand(1, self.input_shape[0], self.input_shape[1], self.input_shape[2])
        features = self.model(pseudo_input)
        return [feat.size()[1:] for feat in features]
class Feature_Extractor_lo(nn.Module):
    def __init__(self, strides, compress_layer, input_shape,oupch=512):
        super(Feature_Extractor_lo, self).__init__()
        self.model = resnet.resnet45lo(strides, compress_layer,oupch=oupch,inpch=input_shape[0])
        self.input_shape = input_shape

    def freezebn(this):
        this.model.freezebn();

    def unfreezebn(this):
        this.model.unfreezebn();

    def forward(self, input):
        features = self.model(input.contiguous())
        return features

    def Iwantshapes(self):
        pseudo_input = torch.rand(1, self.input_shape[0], self.input_shape[1], self.input_shape[2])
        features = self.model(pseudo_input)
        return [feat.size()[1:] for feat in features]

import torch
import torch.nn as nn
import neko_sdk.encoders.ocr_networks.dan.dan_reslens_naive as rescco
import neko_sdk.encoders.ocr_networks.dan.dan_resnet as resnet

class neko_Feature_Extractor_thicc(nn.Module):
    def __init__(self, strides, compress_layer, input_shape,oupch=512):
        super(neko_Feature_Extractor_thicc, self).__init__()
        self.model = resnet.resnet45_thicc(strides, compress_layer,oupch=oupch,inpch=input_shape[0])
        self.input_shape = input_shape

    def forward(self, input):
        features = self.model(input)
        return features

    def freezebn(this):
        this.model.freezebn();

    def unfreezebn(this):
        this.model.unfreezebn();

    def Iwantshapes(self):
        pseudo_input = torch.rand(1, self.input_shape[0], self.input_shape[1], self.input_shape[2])
        features = self.model(pseudo_input)
        return [feat.size()[1:] for feat in features]

class neko_cco_Feature_Extractor(nn.Module):
    def __init__(self, strides, compress_layer, input_shape,hardness=2,oupch=512,expf=1):
        super(neko_cco_Feature_Extractor, self).__init__()
        self.model = rescco.res_naive_lens45(strides, compress_layer,hardness,oupch=oupch,inpch=input_shape[0],expf=expf)
        self.input_shape = input_shape
    def freezebn(this):
        this.model.freezebn();
    def unfreezebn(this):
        this.model.unfreezebn();
    def forward(self, input,debug=False):
        features,grid = self.model(input)
        if debug:
            return features,grid;
        return features

    def Iwantshapes(self):
        pseudo_input = torch.rand(1, self.input_shape[0], self.input_shape[1], self.input_shape[2])
        features,grio = self.model(pseudo_input)
        return [feat.size()[1:] for feat in features]

class neko_cco_Feature_Extractor_thicc(nn.Module):
    def __init__(self, strides, compress_layer, input_shape,hardness=2,oupch=512,expf=1.5):
        super(neko_cco_Feature_Extractor_thicc, self).__init__()
        self.model = rescco.res_naive_lens45_thicc(strides, compress_layer,hardness,oupch=oupch,inpch=input_shape[0])
        self.input_shape = input_shape
    def freezebn(this):
        this.model.freezebn();
    def unfreezebn(this):
        this.model.unfreezebn();
    def forward(self, input,debug=False):
        features,grid = self.model(input)
        if debug:
            return features,grid;
        return features

    def Iwantshapes(self):
        pseudo_input = torch.rand(1, self.input_shape[0], self.input_shape[1], self.input_shape[2])
        features,grio = self.model(pseudo_input)
        return [feat.size()[1:] for feat in features]

class neko_cco_se_Feature_Extractor(nn.Module):
    def __init__(self, strides, compress_layer, input_shape,hardness=2,oupch=512):
        super(neko_cco_se_Feature_Extractor, self).__init__()
        self.model = rescco.res_naive_lens45SE(strides, compress_layer,hardness,oupch=oupch,inpch=input_shape[0])
        self.input_shape = input_shape

    def forward(self, input,debug=False):
        features,grid = self.model(input)
        if debug:
            return features,grid;
        return features

    def Iwantshapes(self):
        pseudo_input = torch.rand(1, self.input_shape[0], self.input_shape[1], self.input_shape[2])
        features,grio = self.model(pseudo_input)
        return [feat.size()[1:] for feat in features]


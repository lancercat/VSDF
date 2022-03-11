import torch
import torch.nn as nn
import neko_2021_mjt.modulars.dan.dan_reslens_naive_std as rescco_std
import neko_2021_mjt.modulars.dan.dan_reslens_naive_sd as rescco_sd



class neko_cco_Feature_Extractor_thicc_std(nn.Module):
    def __init__(self, strides, compress_layer, input_shape,hardness=2,oupch=512):
        super(neko_cco_Feature_Extractor_thicc_std, self).__init__()
        self.model = rescco_std.res_naive_lens45_std_thicc(strides, compress_layer,hardness,oupch=oupch,inpch=input_shape[0])
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


class neko_cco_Feature_Extractor_thicc_stdlr(nn.Module):
    def __init__(self, strides, compress_layer, input_shape,hardness=2,oupch=512):
        super(neko_cco_Feature_Extractor_thicc_stdlr, self).__init__()
        self.model = rescco_std.res_naive_lens45_stdlr_thicc(strides, compress_layer,hardness,oupch=oupch,inpch=input_shape[0])
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

class neko_cco_Feature_Extractor_thicc_sd(nn.Module):
    def __init__(self, strides, compress_layer, input_shape,hardness=2,oupch=512):
        super(neko_cco_Feature_Extractor_thicc_sd, self).__init__()
        self.model = rescco_sd.res_naive_lens45_sd_thicc(strides, compress_layer,hardness,oupch=oupch,inpch=input_shape[0])
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


class neko_cco_Feature_Extractor_thicc_sdlr(nn.Module):
    def __init__(self, strides, compress_layer, input_shape,hardness=2,oupch=512,expf=1):
        super(neko_cco_Feature_Extractor_thicc_sdlr, self).__init__()
        self.model = rescco_sd.res_naive_lens45_sdlr_thicc(strides, compress_layer,hardness,oupch=oupch,inpch=input_shape[0])
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

if __name__ == '__main__':
    a=neko_cco_Feature_Extractor_thicc_sd([(1,1), (2,2), (1,1), (2,2), (1,1), (1,1)],False,[3,32,128]).cuda();
    a.cuda()
    s = a(torch.rand([64, 3, 32, 128]).cuda())
    print(s);
    pass;

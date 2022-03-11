import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

def conv1x1(in_planes,out_planes,stride=1):
    return nn.Conv2d(in_planes,out_planes,kernel_size =1,stride =stride,bias=False)
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class \
        BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out

class dan_ResNet(nn.Module):
    def freezebn(this):
        for m in this.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                if this.freeze_bn_affine:
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False

    def unfreezebn(this):
        for m in this.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.train();
                if this.freeze_bn_affine:
                    m.weight.requires_grad = True
                    m.bias.requires_grad = True

    def __init__(self, block, layers, strides, compress_layer=True,inpch=1,oupch=512,frac=1.0):
        self.inplanes = int(32*frac)
        super(dan_ResNet, self).__init__()
        self.conv1 = nn.Conv2d(inpch, int(32*frac), kernel_size=3, stride=strides[0], padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(int(32*frac))
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, int(32*frac), layers[0],stride=strides[1])
        self.layer2 = self._make_layer(block, int(64*frac), layers[1], stride=strides[2])
        self.layer3 = self._make_layer(block, int(128*frac), layers[2], stride=strides[3])
        self.layer4 = self._make_layer(block, int(256*frac), layers[3], stride=strides[4])
        if compress_layer:
            self.layer5 = self._make_layer(block, int(512*frac), layers[4], stride=strides[5])
        else:
            self.layer5 = self._make_layer(block, oupch, layers[4], stride=strides[5])
        self.freeze_bn_affine=False;
        self.compress_layer = compress_layer        
        if compress_layer:
            # for handwritten
            self.layer6 = nn.Sequential(
                nn.Conv2d(512, oupch, kernel_size=(3, 1), padding=(0, 0), stride=(1, 1)),
                nn.BatchNorm2d(oupch),
                nn.ReLU(inplace = True))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, multiscale = False):
        out_features = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        tmp_shape = x.size()[2:]
        x = self.layer1(x)
        if x.size()[2:] != tmp_shape:
            tmp_shape = x.size()[2:]
            out_features.append(x)
        x = self.layer2(x)
        if x.size()[2:] != tmp_shape:
            tmp_shape = x.size()[2:]
            out_features.append(x)
        x = self.layer3(x)
        if x.size()[2:] != tmp_shape:
            tmp_shape = x.size()[2:]
            out_features.append(x)
        x = self.layer4(x)
        if x.size()[2:] != tmp_shape:
            tmp_shape = x.size()[2:]
            out_features.append(x)
        x = self.layer5(x)
        if not self.compress_layer:
            out_features.append(x)
        else:
            if x.size()[2:] != tmp_shape:
                tmp_shape = x.size()[2:]
                out_features.append(x)
            x = self.layer6(x)
            out_features.append(x)
        return out_features

    def forward_dump(self, x, multiscale=False):
        out_features = []
        all_inters={};
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        all_inters["l0"]=x.detach().cpu();
        tmp_shape = x.size()[2:]
        x = self.layer1(x)
        if x.size()[2:] != tmp_shape:
            tmp_shape = x.size()[2:]
            out_features.append(x)
        all_inters["l1"] = x.detach().cpu();

        x = self.layer2(x)
        if x.size()[2:] != tmp_shape:
            tmp_shape = x.size()[2:]
            out_features.append(x)
        all_inters["l2"] = x.detach().cpu();

        x = self.layer3(x)
        if x.size()[2:] != tmp_shape:
            tmp_shape = x.size()[2:]
            out_features.append(x)
        all_inters["l3"] = x.detach().cpu();

        x = self.layer4(x)
        if x.size()[2:] != tmp_shape:
            tmp_shape = x.size()[2:]
            out_features.append(x)
        all_inters["l4"] = x.detach().cpu();

        x = self.layer5(x)
        all_inters["l5"] = x.detach().cpu();

        if not self.compress_layer:
            out_features.append(x)
        else:
            if x.size()[2:] != tmp_shape:
                tmp_shape = x.size()[2:]
                out_features.append(x)
            x = self.layer6(x)
            out_features.append(x.contiguous())
        return out_features,all_inters

def resnet45(strides, compress_layer,oupch=512,inpch=1):
    model = dan_ResNet(BasicBlock, [3, 4, 6, 6, 3], strides, compress_layer,oupch=oupch,inpch=inpch,frac=1)
    return model
def  resnet45_thicc(strides, compress_layer,oupch=512,inpch=1):
    model = dan_ResNet(BasicBlock, [3, 4, 6, 6, 3], strides, compress_layer,oupch=oupch,inpch=inpch,frac=1.5)
    return model
if __name__ == '__main__':
    import torch;
    import pthflops
    strides=[(1,1), (2,2), (1,1), (2,2), (1,1), (1,1)]
    net=resnet45(strides,None);

    a=torch.rand([1,1,32,128]);
    aa=net(a);

    macs, params = pthflops.count_ops(net, a)
    print(macs);
    pass;

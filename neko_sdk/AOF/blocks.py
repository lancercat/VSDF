from neko_sdk.AOF.neko_lens import neko_lens_fuse;
from torch import nn


def conv1x1(in_planes,out_planes,stride=1):
    return nn.Conv2d(in_planes,out_planes,kernel_size =1,stride =stride,bias=False)
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlockNoLens(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlockNoLens, self).__init__()
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
        return out,None;

class neko_LensBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):

        super(neko_LensBlock, self).__init__()
        if inplanes!=planes:
            self.residule_hack=conv1x1(inplanes, planes);
        else:
            self.residule_hack=None;
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.fuser = neko_lens_fuse(planes);
        self.stride = stride

    def forward(self, x):
        residule=x;
        if(self.residule_hack is not None):
            residule=self.residule_hack(x);
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out,grid = self.fuser(residule, out)
        out = self.relu(out)
        return out,grid;
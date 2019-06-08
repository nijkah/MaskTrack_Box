import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
from torch.nn import functional as F
import numpy as np

from models.aspp import build_aspp

affine_par = True

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, affine = affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, affine = affine_par)
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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1,  dilation_ = 1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False) # change
        self.bn1 = nn.BatchNorm2d(planes,affine = affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        padding = 1
        if dilation_ > 1:
            padding = dilation_
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, # change
                               padding=padding, bias=False, dilation = dilation_)
        self.bn2 = nn.BatchNorm2d(planes,affine = affine_par)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine = affine_par)
        for i in self.bn3.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride



    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Classifier_Module(nn.Module):

    def __init__(self,dilation_series,padding_series,NoLabels):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation,padding in zip(dilation_series,padding_series):
            self.conv2d_list.append(nn.Conv2d(2048,NoLabels,kernel_size=3,stride=1, padding = padding, dilation = dilation,bias = True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)


    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list)-1):
            out += self.conv2d_list[i+1](x)
        return out



class ResNet(nn.Module):
    def __init__(self, block, layers,NoLabels, in_channel=3):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64,affine = affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True) # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation__ = 2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation__ = 4)
        #self.layer5 = self._make_pred_layer(Classifier_Module, [6,12,18,24],[6,12,18,24], NoLabels)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        #        for i in m.parameters():
        #            i.requires_grad = False

    def _make_layer(self, block, planes, blocks, stride=1,dilation__ = 1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation__ == 2 or dilation__ == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion,affine = affine_par),
            )
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride,dilation_=dilation__, downsample = downsample ))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,dilation_=dilation__))

        return nn.Sequential(*layers)
    def _make_pred_layer(self,block, dilation_series, padding_series,NoLabels):
        return block(dilation_series,padding_series,NoLabels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        low_level_feat = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        #x = self.layer5(x)

        return x, low_level_feat

class ResNet_ms(nn.Module):
    def __init__(self, block, layers,NoLabels, in_channel=3):
        self.inplanes = 64
        super(ResNet_ms, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64,affine = affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True) # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation__ = 2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation__ = 4)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        #        for i in m.parameters():
        #            i.requires_grad = False

    def _make_layer(self, block, planes, blocks, stride=1,dilation__ = 1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation__ == 2 or dilation__ == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion,affine = affine_par),
            )

        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride,dilation_=dilation__, downsample = downsample ))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation_=dilation__))

        return nn.Sequential(*layers)
    def _make_pred_layer(self,block, dilation_series, padding_series,NoLabels):
        return block(dilation_series,padding_series,NoLabels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x0_5 = self.relu(x)
        x0_25 = self.maxpool(x0_5)
        x0_25 = self.layer1(x0_25)
        x = self.layer2(x0_25)
        x = self.layer3(x)
        x = self.layer4(x)

        return [x0_5, x0_25, x]


class MS_Deeplab_ms(nn.Module):
    def __init__(self,block,NoLabels):
        super(MS_Deeplab_ms,self).__init__()
        self.Scale = ResNet(block,[3, 4, 23, 3],NoLabels, in_channel=3)   #changed to fix #4 

    def forward(self,x):
        input_size = x.size()[2]
        self.interp1 = nn.Upsample(size = (  int(h*0.75)+1,  int(w*0.75)+1  ), mode='bilinear')
        self.interp2 = nn.Upsample(size = (  int(h*0.5)+1,   int(w*0.5)+1   ), mode='bilinear')
        out = []
        x2 = self.interp1(x)
        x3 = self.interp2(x)
        out.append(self.Scale(x))	# for original scale
        out.append(self.interp3(self.Scale(x2)))	# for 0.75x scale
        out.append(self.Scale(x3))	# for 0.5x scale

        x2Out_interp = out[1]
        x3Out_interp = self.interp3(out[2])
        temp1 = torch.max(out[0],x2Out_interp)
        out.append(torch.max(temp1,x3Out_interp))
        return out

class MS_Deeplab_ms(nn.Module):
    def __init__(self,block,NoLabels):
        super(MS_Deeplab_ms,self).__init__()
        self.Scale = ResNet_ms(block,[3, 4, 23, 3],NoLabels, in_channel=3)   #changed to fix #4 
        self.aspp = build_aspp(output_stride=16)

        self.branch = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        
        self.fuse = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 2048, kernel_size=1),
            nn.BatchNorm2d(2048),
            nn.ReLU())

        self.template_refine= nn.Sequential(
                #nn.Conv2d(48+256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.Conv2d(2048, 256, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True), # change
                nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU())

        self.template_fuse = nn.Sequential(
                nn.Conv2d(128+128, 256, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True), # change
                nn.Conv2d(256, 2048, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(2048),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True))



        self.refine= nn.Sequential(
                nn.Conv2d(256+256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU())

        self.predict = nn.Sequential(
                #nn.Conv2d(48+256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.Conv2d(128+64, 128, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, NoLabels, kernel_size=1))

        

        #self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406, -0.329]).view(1,4,1,1))
        #self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225, 0.051]).view(1,4,1,1))
        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1,3,1,1))

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def set_template(self, x, mask):
        x = (x - self.mean) / self.std
        ll, low_level_feat, out = self.Scale(x)
        out = self.template_refine(out)
        mask = F.interpolate(mask, size=out.shape[2:])
        branch = out * mask
        out = torch.cat([out, branch], 1)
        out = self.template_fuse(out)

        #branch_feature = self.branch(out)
        #mask_feature = branch_feature * mask
        #fused_feature = torch.cat([branch_feature, mask_feature], 1)
        #fused_feature = self.fuse(fused_feature)
        #out = out + fused_feature
        template_feature = F.max_pool2d(out, kernel_size=out.shape[2:])

        return template_feature

    def forward(self, x, mask, target, box):
        template_feature = self.set_template(target, box)

        x = (x - self.mean) / self.std
        ll, low_level_feat, out,  = self.Scale(x)
        mask = F.interpolate(mask, size=out.shape[2:])

        branch_feature = self.branch(out)
        mask_feature = branch_feature * mask
        fused_feature = torch.cat([branch_feature, mask_feature], 1)
        fused_feature = self.fuse(fused_feature)
        out = out + fused_feature
        out = out * template_feature

        out = self.aspp(out)
        #branch_feature = self.branch(low_level_feat)
        
        #mask = F.interpolate(mask, size=branch_feature.shape[2:])
        #mask_feature = branch_feature * mask
        #fused_feature = torch.cat([branch_feature, mask_feature], 1)
        #fused_feature = self.fuse(fused_feature)
        #branch_feature = branch_feature + fused_feature

        out = F.interpolate(out, size=low_level_feat.shape[2:])
        out = torch.cat([out, low_level_feat], 1)
        out = self.refine(out)
        out = F.interpolate(out, size=ll.shape[2:])
        out = torch.cat([out, ll], 1)
        out = self.predict(out)
        #out = F.interpolate(out, size=(161, 161))

        return out

def Res_Deeplab(NoLabels=21):
    model = MS_Deeplab_ms(Bottleneck,NoLabels)
    return model

def Res_Deeplab_4chan(NoLabels=21):
    model = MS_Deeplab_ms(Bottleneck, NoLabels)
    return model


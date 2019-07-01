import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
from torch.nn import functional as F
import numpy as np

from models.backbone import build_backbone
from models.aspp import build_aspp

class Siam_Deeplab(nn.Module):
    def __init__(self, NoLabels, pretrained=False):
        super(Siam_Deeplab, self).__init__()
        self.backbone= build_backbone('resnet_ms', in_channel=3, pretrained=pretrained)
        self.aspp = build_aspp(output_stride=16)

        self.conv1_1 = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU())

        self.branch = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        
        self.fuse = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.fuse2 = nn.Sequential(
            nn.Conv2d(256+128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())

        #self.template_refine= nn.Sequential(
        #        #nn.Conv2d(48+256, 256, kernel_size=3, stride=1, padding=1, bias=False),
        #        nn.Conv2d(1024, 256, kernel_size=7, stride=2, padding=3, bias=False),
        #        nn.BatchNorm2d(256),
        #        nn.ReLU(),
        #        #nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True), # change
        #        nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #        nn.BatchNorm2d(128),
        #        nn.ReLU())

        self.template_fuse = nn.Sequential(
                nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                #nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True), # change
                nn.Conv2d(512, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU())
                #nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True))

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

    def fuse_mask(self, x, mask, is_ref=False):
        mask = F.interpolate(mask, size=x.shape[2:])
        branch = x * mask
        branch = branch + x
        out = torch.cat([x, branch], 1)
        out = self.template_fuse(out) if is_ref else self.fuse(out)

        return out
        
    def set_template(self, x, mask):
        x = (x - self.mean) / self.std
        ll, low_level_feat, out = self.backbone(x)
        out = self.conv1_1(out)
        out = self.fuse_mask(out, mask, is_ref=True)
        template_feature = F.max_pool2d(out, kernel_size=out.shape[2:])
        
        return template_feature

    def forward(self, x, mask, target, box):
        template_feature = self.set_template(target, box)

        x = (x - self.mean) / self.std
        ll, low_level_feat, out_o = self.backbone(x)
        out = self.conv1_1(out_o)
        out = self.fuse_mask(out, mask)

        #out = out + fused_feature
        diff = torch.abs(out - template_feature)
        out = torch.cat([out, diff], 1)
        out_o = self.aspp(out_o)
        out = torch.cat([out_o, out], 1)
        out = self.fuse2(out)
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

    def get_1x_lr_params_NOscale(self):
        """
        This generator returns all the parameters of the net except for 
        the last classification layer. Note that for each batchnorm layer, 
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return 
        any batchnorm parameter
        """
        b = []

        b.append(self.backbone.conv1)
        b.append(self.backbone.bn1)
        b.append(self.backbone.layer1)
        b.append(self.backbone.layer2)
        b.append(self.backbone.layer3)
        b.append(self.backbone.layer4)
        b.append(self.conv1_1)
        b.append(self.branch)
        b.append(self.fuse)
        b.append(self.template_fuse)
        b.append(self.refine)

        
        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj+=1
                    if k.requires_grad:
                        yield k

    def get_10x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """

        b = []
        b.append(self.aspp.parameters())
        #b.append(self.template_refine.parameters())
        b.append(self.predict.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i


def build_siam_Deeplab(NoLabels=21, pretrained=False):
    model = Siam_Deeplab(NoLabels, pretrained)
    return model


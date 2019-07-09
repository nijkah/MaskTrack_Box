import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
from torch.nn import functional as F
import numpy as np

from models.backbone import build_backbone

def _make_pred_layer(block, dilation_series, padding_series,NoLabels):
    return block(dilation_series,padding_series,NoLabels)

class Classifier_Module(nn.Module):

    def __init__(self,dilation_series,padding_series,NoLabels):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation,padding in zip(dilation_series,padding_series):
            self.conv2d_list.append(nn.Conv2d(2048,NoLabels,kernel_size=3,stride=1, padding =padding, dilation = dilation,bias = True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)


    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list)-1):
            out += self.conv2d_list[i+1](x)
        return out

class MS_Deeplab(nn.Module):
    def __init__(self, NoLabels, pretrained=False):
        super(MS_Deeplab,self).__init__()
        self.Scale = build_backbone('resnet', in_channel=4, pretrained=pretrained)
        self.classifier = _make_pred_layer(Classifier_Module, [6,12,18,24],[6,12,18,24], NoLabels)

        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406, 0]).view(1,4,1,1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225, 0.358]).view(1,4,1,1))

    
    def forward(self, x):
        x = (x - self.mean) / self.std
        out = self.Scale(x)
        out = self.classifier(out)

        return out

    def get_1x_lr_params_NOscale(self):
        """
        This generator returns all the parameters of the net except for 
        the last classification layer. Note that for each batchnorm layer, 
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return 
        any batchnorm parameter
        """
        b = []

        b.append(self.Scale.conv1)
        b.append(self.Scale.bn1)
        b.append(self.Scale.layer1)
        b.append(self.Scale.layer2)
        b.append(self.Scale.layer3)
        b.append(self.Scale.layer4)

        
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
        b.append(self.classifier.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i

def build_Deeplab(NoLabels=2, pretrained=False):
    model = MS_Deeplab(NoLabels, pretrained=pretrained)
    return model


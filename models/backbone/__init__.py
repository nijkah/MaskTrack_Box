from models.backbone import resnet

def build_backbone(backbone, in_channel, pretrained=True):
    ms = False
    if backbone == 'resnet_ms':
        ms = True

    return resnet.ResNet101(in_channel, ms, pretrained)

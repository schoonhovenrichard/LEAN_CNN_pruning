from msd_pytorch.msd_model import (MSDModel)
from torch.autograd import Variable
from model_backbones.resnet import resnet
from model_backbones.resnet.resnet_utils.utils_resnet import IntermediateLayerGetter
from model_backbones.resnet.resnet_utils.fcn import FCN, FCNHead
from model_backbones.resnet.resnet_utils.deeplabv3 import DeepLabHead, DeepLabV3
from torch.utils.model_zoo import load_url as load_state_dict_from_url
import torch as t

from collections import OrderedDict
import warnings
import sys
import numpy as np
import torch.nn as nn

def _segm_resnet(name, backbone_name, num_classes, aux, pretrained_backbone=True, c_in=3):
    backbone = resnet.__dict__[backbone_name](
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=[False, True, True],
        c_in=c_in)

    return_layers = {'layer4': 'out'}
    if aux:
        return_layers['layer3'] = 'aux'
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    aux_classifier = None
    if aux:
        inplanes = 1024
        aux_classifier = FCNHead(inplanes, num_classes)

    model_map = {
        'deeplab': (DeepLabHead, DeepLabV3),
        'fcn': (FCNHead, FCN),
    }
    inplanes = 2048
    classifier = model_map[name][0](inplanes, num_classes)
    base_model = model_map[name][1]

    model = base_model(backbone, classifier, aux_classifier)
    return model

def fcn_resnet50(pretrained=False, progress=True,
                 num_classes=21, aux_loss=None, **kwargs):
    """Constructs a Fully-Convolutional Network model with a ResNet-50 backbone.
    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    if pretrained:
        aux_loss = True
    model = _segm_resnet("fcn", "resnet50", num_classes, aux_loss, **kwargs)
    if pretrained:
        arch = 'fcn_resnet50_coco'
        model_url = model_urls[arch]
        if model_url is None:
            raise NotImplementedError('pretrained {} is not supported as of now'.format(arch))
        else:
            state_dict = load_state_dict_from_url(model_url, progress=progress)
            model.load_state_dict(state_dict)
    return model

def fcn_resnet101(pretrained=False, progress=True,
                  num_classes=21, aux_loss=None, **kwargs):
    """Constructs a Fully-Convolutional Network model with a ResNet-101 backbone.
    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    if pretrained:
        aux_loss = True
    model = _segm_resnet("fcn", "resnet101", num_classes, aux_loss, **kwargs)
    if pretrained:
        arch = 'fcn_resnet101_coco'
        model_url = model_urls[arch]
        if model_url is None:
            raise NotImplementedError('pretrained {} is not supported as of now'.format(arch))
        else:
            state_dict = load_state_dict_from_url(model_url, progress=progress)
            model.load_state_dict(state_dict)
    return model


def deeplabv3_resnet50(pretrained=False, progress=True,
                       num_classes=21, aux_loss=None, **kwargs):
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.
    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    if pretrained:
        aux_loss = True
    model = _segm_resnet("deeplab", "resnet50", num_classes, aux_loss, **kwargs)
    if pretrained:
        arch = 'deeplabv3_resnet50_coco'
        model_url = model_urls[arch]
        if model_url is None:
            raise NotImplementedError('pretrained {} is not supported as of now'.format(arch))
        else:
            state_dict = load_state_dict_from_url(model_url, progress=progress)
            model.load_state_dict(state_dict)
    return model

def deeplabv3_resnet101(pretrained=False, progress=True,
                        num_classes=21, aux_loss=None, **kwargs):
    """Constructs a DeepLabV3 model with a ResNet-101 backbone.
    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    if pretrained:
        aux_loss = True
    model = _segm_resnet("deeplab", "resnet101", num_classes, aux_loss, **kwargs)
    if pretrained:
        arch = 'deeplabv3_resnet101_coco'
        model_url = model_urls[arch]
        if model_url is None:
            raise NotImplementedError('pretrained {} is not supported as of now'.format(arch))
        else:
            state_dict = load_state_dict_from_url(model_url, progress=progress)
            model.load_state_dict(state_dict)
    return model

class ResNet50AvgSegmentationModel(MSDModel):
    def __init__(self, c_in, num_labels, pretrained=False):
        # We don't support 3d segmentation yet.
        # Allow supplying a list of labels instead of just the number
        # of labels.
        if isinstance(num_labels, list):
            c_out = 1
            self.labels = num_labels
        else:
            self.labels = range(num_labels)
            c_out = 1

        # Initialize resnet network.
        super().__init__(c_in, num_labels, 1, 1, [1,2,3,4,5,6,7,8,9,10])
        # LogSoftmax + NLLLoss is equivalent to a Softmax activation
        # with Cross-entropy loss.
        self.criterion = nn.NLLLoss()

        # Make Resnet
        self.msd = fcn_resnet50(num_classes=num_labels, pretrained_backbone=pretrained, c_in=c_in)

        avgpool = nn.Conv2d(64, 64, 3, padding=1, stride=2, bias=False)
        avgpool.weight.data.fill_(0)
        for c in range(avgpool.weight.size()[0]):
            avgpool.weight.data[c,c,:,:].fill_(1/float(9))
        #avgpool.bias.data.fill_(0)
        for param in avgpool.parameters():
             param.requires_grad = False
        self.msd.backbone.maxpool = avgpool

        # Initialize network
        #net_trained = nn.Sequential(self.msd, nn.LogSoftmax(dim=1))
        self.net = nn.Sequential(self.scale_in, self.msd)
        self.net.cuda()

        # Train all parameters apart from self.scale_in.
        self.init_optimizer(self.msd)

    def set_normalization(self, dataloader):
        mean = 0
        square = 0
        for (data_in, _) in dataloader:
            mean += data_in.mean()
            square += data_in.pow(2).mean()

        mean /= len(dataloader)
        square /= len(dataloader)
        std = np.sqrt(square - mean ** 2)

        # The input data should be roughly normally distributed after
        # passing through net_fixed.
        self.scale_in.bias.data.fill_(- mean / std)
        self.scale_in.weight.data.fill_(1 / std)

    def set_target(self, data):
        # relabel if necessary:
        target = data.clone()
        if self.labels:
            for i, label in enumerate(self.labels):
                target[data == label] = i

        # The class labels must be of long data type
        target = target.long()
        # The NLLLoss does not accept a channel dimension. So we
        # squeeze the target.
        target = target.squeeze(1)
        # The class labels must reside on the GPU
        target = target.cuda()
        self.target = Variable(target)

    def load(self, path, strict=True, expanded=False):
        state = t.load(path)

        if 'state_dict' in state:
            state_dict = state['state_dict']
        else:
            state_dict = state
        if strict:
            self.net.load_state_dict(state_dict, strict=strict)
        else:
            model_dict = self.net.state_dict()
            new_state_dict = OrderedDict()
            matched_keys, discarded_keys = [], []

            for k, v in state_dict.items():
                if k in model_dict and model_dict[k].size() == v.size():
                    new_state_dict[k] = v
                    matched_keys.append(k)
                else:
                    discarded_keys.append(k)

            model_dict.update(new_state_dict)
            self.net.load_state_dict(state_dict, strict=strict)

            if len(matched_keys) == 0:
                warnings.warn(
                    'The pretrained weights cannot be loaded, '
                    'please check the key names manually '
                    '(** ignored and continue **)'
                )
            else:
                print(
                    'Successfully loaded pretrained weights.'
                )
                if len(discarded_keys) > 0:
                    print(
                        '** The following keys are discarded '
                        'due to unmatched keys or layer size: {}'.
                        format(discarded_keys)
                    )
        self.optimizer.load_state_dict(state["optimizer"])
        self.net.cuda()

        epoch = state["epoch"]
        return epoch

    def simple_load(self, path):
        state = t.load(path)

        self.net.load_state_dict(state["state_dict"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.net.cuda()
        #self.net.eval() # NOTE: User has to manually call model.train() or model.eval() themselves!

        epoch = state["epoch"]

        return epoch


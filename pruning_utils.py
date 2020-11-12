### CVPR 2021 Submission #8167. Confidential review copy. Do not distribute.

import torch

def get_default_mask(modul, nam):
    r"""Retrieve the current pruned mask of a module.
    """
    orig = getattr(modul, nam)
    try:
        default_mask = getattr(modul, nam + "_mask").detach().clone(memory_format=torch.contiguous_format)
    except Exception as e:
        default_mask = torch.ones_like(orig)
    return default_mask

def fraction_pruned_convs_ResNet50(model):
    r"""Return the fraction of pruned convs of ResNet50 model.
    """
    conv_masks = get_conv_masks_ResNet50(model)
    tot_conv = 0
    pp = 0
    for mask in conv_masks:
        tot_conv += mask.size()[0] * mask.size()[1]
        count = mask.sum(axis=-1)
        count = count.sum(axis=-1)
        pp += (count == 0).sum().item()
    return (pp, tot_conv, pp/float(tot_conv))

def fraction_pruned_convs_MSD(model):
    r"""Return the fraction of pruned convs of MSD model.
    """
    conv_masks = get_conv_masks_MSD(model)
    tot_conv = 0
    pp = 0
    for mask in conv_masks:
        if mask.size()[-1] == 1:
            tot_conv += mask.size()[0] * mask.size()[1]
            count = mask.sum(axis=-1)
            pp += (count == 0).sum().item()
        else:
            tot_conv += mask.size()[0] * mask.size()[1]
            count = mask.sum(axis=-1)
            count = count.sum(axis=-1)
            pp += (count == 0).sum().item()
    return (pp, tot_conv, pp/float(tot_conv))

def fraction_pruned_convs_MSD3x3(model):
    r"""Return the fraction of pruned convs of MSD model,
        excluding the final layer.
    """
    conv_masks = get_conv_masks_MSD3x3(model)
    tot_conv = 0
    pp = 0
    for mask in conv_masks:
        tot_conv += mask.size()[0] * mask.size()[1]
        count = mask.sum(axis=-1)
        count = count.sum(axis=-1)
        pp += (count == 0).sum().item()
    return (pp, tot_conv, pp/float(tot_conv))

def pruned_before_ResNet50(model):
    r"""Check if FCN-ResNet50 has been pruned before
        by checking the buffer for masks.
    """
    pruned_before = False
    mod1 = model.msd.backbone
    mod2 = model.msd.classifier
    for x in mod1.named_buffers():
        if "mask" in x[0]:
            pruned_before = True
            break
    for x in mod2.named_buffers():
        if "mask" in x[0]:
            pruned_before = True
            break
    return pruned_before

def pruned_before_MSD(model):
    r"""Check if MS-D model has been pruned before
        by checking the buffer for masks.
    """
    pruned_before = False
    mod = model.msd.msd_block
    for x in mod.named_buffers():
        if "mask" in x[0]:
            pruned_before = True
            break
    return pruned_before


### Functions to get conv-layers, masks etc.:

def get_convs_MSD(model):
    r"""Retrieve list of convolutional layers in MS-D network.
    """
    mod = model.msd.msd_block
    convolutions = []
    for k in range(model.depth):
        wname ="weight"+str(k)
        convolutions.append((mod, wname))
    convolutions.append((model.msd.final_layer.linear, "weight"))
    return convolutions

def get_conv_masks_MSD(model):
    r"""Get list of convolution masks for MS-D network.
    """
    mod = model.msd.msd_block
    convolution_masks = []
    for k in range(model.depth):
        wname ="weight"+str(k)+"_mask"
        convolution_masks.append(getattr(mod, wname))
    convolution_masks.append(model.msd.final_layer.linear.weight_mask)
    return convolution_masks

def get_conv_masks_MSD3x3(model):
    r"""Get list of convolution masks for MS-D network,
        excluding the final layer.
    """
    mod = model.msd.msd_block
    convolution_masks = []
    for k in range(model.depth):
        wname ="weight"+str(k)+"_mask"
        convolution_masks.append(getattr(mod, wname))
    return convolution_masks

def get_convs_ResNet50(model):
    r"""Get list of convolution layers for FCN-ResNet50.
    """
    mod1 = model.msd.backbone
    mod2 = model.msd.classifier
    convolutions = [
        (mod1.conv1, 'weight'),
        (mod1.layer1[0].conv1, 'weight'),
        (mod1.layer1[0].conv2, 'weight'),
        (mod1.layer1[0].conv3, 'weight'),
        (mod1.layer1[0].downsample[0], 'weight'),
        (mod1.layer1[1].conv1, 'weight'),
        (mod1.layer1[1].conv2, 'weight'),
        (mod1.layer1[1].conv3, 'weight'),
        (mod1.layer1[2].conv1, 'weight'),
        (mod1.layer1[2].conv2, 'weight'),
        (mod1.layer1[2].conv3, 'weight'),
        (mod1.layer2[0].conv1, 'weight'),
        (mod1.layer2[0].conv2, 'weight'),
        (mod1.layer2[0].conv3, 'weight'),
        (mod1.layer2[0].downsample[0], 'weight'),
        (mod1.layer2[1].conv1, 'weight'),
        (mod1.layer2[1].conv2, 'weight'),
        (mod1.layer2[1].conv3, 'weight'),
        (mod1.layer2[2].conv1, 'weight'),
        (mod1.layer2[2].conv2, 'weight'),
        (mod1.layer2[2].conv3, 'weight'),
        (mod1.layer2[3].conv1, 'weight'),
        (mod1.layer2[3].conv2, 'weight'),
        (mod1.layer2[3].conv3, 'weight'),
        (mod1.layer3[0].conv1, 'weight'),
        (mod1.layer3[0].conv2, 'weight'),
        (mod1.layer3[0].conv3, 'weight'),
        (mod1.layer3[0].downsample[0], 'weight'),
        (mod1.layer3[1].conv1, 'weight'),
        (mod1.layer3[1].conv2, 'weight'),
        (mod1.layer3[1].conv3, 'weight'),
        (mod1.layer3[2].conv1, 'weight'),
        (mod1.layer3[2].conv2, 'weight'),
        (mod1.layer3[2].conv3, 'weight'),
        (mod1.layer3[3].conv1, 'weight'),
        (mod1.layer3[3].conv2, 'weight'),
        (mod1.layer3[3].conv3, 'weight'),
        (mod1.layer3[4].conv1, 'weight'),
        (mod1.layer3[4].conv2, 'weight'),
        (mod1.layer3[4].conv3, 'weight'),
        (mod1.layer3[5].conv1, 'weight'),
        (mod1.layer3[5].conv2, 'weight'),
        (mod1.layer3[5].conv3, 'weight'),
        (mod1.layer4[0].conv1, 'weight'),
        (mod1.layer4[0].conv2, 'weight'),
        (mod1.layer4[0].conv3, 'weight'),
        (mod1.layer4[0].downsample[0], 'weight'),
        (mod1.layer4[1].conv1, 'weight'),
        (mod1.layer4[1].conv2, 'weight'),
        (mod1.layer4[1].conv3, 'weight'),
        (mod1.layer4[2].conv1, 'weight'),
        (mod1.layer4[2].conv2, 'weight'),
        (mod1.layer4[2].conv3, 'weight'),
        (mod2[0], 'weight'),
        (mod2[4], 'weight')
    ]
    return convolutions

def get_batchnorms_ResNet50(model):
    r"""Get list of batchnormalization layer for FCN-ResNet50.
    """
    mod1 = model.msd.backbone
    mod2 = model.msd.classifier
    batchnorms = [
        mod1.bn1,
        mod1.layer1[0].bn1,
        mod1.layer1[0].bn2,
        mod1.layer1[0].bn3,
        mod1.layer1[0].downsample[1],
        mod1.layer1[1].bn1,
        mod1.layer1[1].bn2,
        mod1.layer1[1].bn3,
        mod1.layer1[2].bn1,
        mod1.layer1[2].bn2,
        mod1.layer1[2].bn3,
        mod1.layer2[0].bn1,
        mod1.layer2[0].bn2,
        mod1.layer2[0].bn3,
        mod1.layer2[0].downsample[1],
        mod1.layer2[1].bn1,
        mod1.layer2[1].bn2,
        mod1.layer2[1].bn3,
        mod1.layer2[2].bn1,
        mod1.layer2[2].bn2,
        mod1.layer2[2].bn3,
        mod1.layer2[3].bn1,
        mod1.layer2[3].bn2,
        mod1.layer2[3].bn3,
        mod1.layer3[0].bn1,
        mod1.layer3[0].bn2,
        mod1.layer3[0].bn3,
        mod1.layer3[0].downsample[1],
        mod1.layer3[1].bn1,
        mod1.layer3[1].bn2,
        mod1.layer3[1].bn3,
        mod1.layer3[2].bn1,
        mod1.layer3[2].bn2,
        mod1.layer3[2].bn3,
        mod1.layer3[3].bn1,
        mod1.layer3[3].bn2,
        mod1.layer3[3].bn3,
        mod1.layer3[4].bn1,
        mod1.layer3[4].bn2,
        mod1.layer3[4].bn3,
        mod1.layer3[5].bn1,
        mod1.layer3[5].bn2,
        mod1.layer3[5].bn3,
        mod1.layer4[0].bn1,
        mod1.layer4[0].bn2,
        mod1.layer4[0].bn3,
        mod1.layer4[0].downsample[1],
        mod1.layer4[1].bn1,
        mod1.layer4[1].bn2,
        mod1.layer4[1].bn3,
        mod1.layer4[2].bn1,
        mod1.layer4[2].bn2,
        mod1.layer4[2].bn3,
        mod2[1],
        None
        ]
    return batchnorms

def get_conv_masks_ResNet50(model):
    r"""Get list of convolution masks for FCN-ResNet50.
    """
    mod1 = model.msd.backbone
    mod2 = model.msd.classifier
    conv_masks = [
        mod1.conv1.weight_mask,
        mod1.layer1[0].conv1.weight_mask,
        mod1.layer1[0].conv2.weight_mask,
        mod1.layer1[0].conv3.weight_mask,
        mod1.layer1[0].downsample[0].weight_mask,
        mod1.layer1[1].conv1.weight_mask,
        mod1.layer1[1].conv2.weight_mask,
        mod1.layer1[1].conv3.weight_mask,
        mod1.layer1[2].conv1.weight_mask,
        mod1.layer1[2].conv2.weight_mask,
        mod1.layer1[2].conv3.weight_mask,
        mod1.layer2[0].conv1.weight_mask,
        mod1.layer2[0].conv2.weight_mask,
        mod1.layer2[0].conv3.weight_mask,
        mod1.layer2[0].downsample[0].weight_mask,
        mod1.layer2[1].conv1.weight_mask,
        mod1.layer2[1].conv2.weight_mask,
        mod1.layer2[1].conv3.weight_mask,
        mod1.layer2[2].conv1.weight_mask,
        mod1.layer2[2].conv2.weight_mask,
        mod1.layer2[2].conv3.weight_mask,
        mod1.layer2[3].conv1.weight_mask,
        mod1.layer2[3].conv2.weight_mask,
        mod1.layer2[3].conv3.weight_mask,
        mod1.layer3[0].conv1.weight_mask,
        mod1.layer3[0].conv2.weight_mask,
        mod1.layer3[0].conv3.weight_mask,
        mod1.layer3[0].downsample[0].weight_mask,
        mod1.layer3[1].conv1.weight_mask,
        mod1.layer3[1].conv2.weight_mask,
        mod1.layer3[1].conv3.weight_mask,
        mod1.layer3[2].conv1.weight_mask,
        mod1.layer3[2].conv2.weight_mask,
        mod1.layer3[2].conv3.weight_mask,
        mod1.layer3[3].conv1.weight_mask,
        mod1.layer3[3].conv2.weight_mask,
        mod1.layer3[3].conv3.weight_mask,
        mod1.layer3[4].conv1.weight_mask,
        mod1.layer3[4].conv2.weight_mask,
        mod1.layer3[4].conv3.weight_mask,
        mod1.layer3[5].conv1.weight_mask,
        mod1.layer3[5].conv2.weight_mask,
        mod1.layer3[5].conv3.weight_mask,
        mod1.layer4[0].conv1.weight_mask,
        mod1.layer4[0].conv2.weight_mask,
        mod1.layer4[0].conv3.weight_mask,
        mod1.layer4[0].downsample[0].weight_mask,
        mod1.layer4[1].conv1.weight_mask,
        mod1.layer4[1].conv2.weight_mask,
        mod1.layer4[1].conv3.weight_mask,
        mod1.layer4[2].conv1.weight_mask,
        mod1.layer4[2].conv2.weight_mask,
        mod1.layer4[2].conv3.weight_mask,
        mod2[0].weight_mask,
        mod2[4].weight_mask
    ]
    return conv_masks

import torch
import msd_pytorch as mp
import numpy as np
from timeit import default_timer as timer
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch_pruning_methods import *
import graph_prune_utils as gru
import networkx as nx
from analysis_utils import get_norm_histogram
from analyze_masks import draw_adjacency_matrix

#from unet_seg_model import UNetSegmentationModel
#from resnet_seg_model import ResNetSegmentationModel

#TODO: Add more docstrings

def get_default_mask(modul, nam):
    orig = getattr(modul, nam)
    try:
        default_mask = getattr(modul, nam + "_mask").detach().clone(memory_format=torch.contiguous_format)
    except Exception as e:
        default_mask = torch.ones_like(orig)
    return default_mask

def pruned_before_ResNet50(model):
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

def get_convs_MSD(model):
    mod = model.msd.msd_block
    convolutions = []
    for k in range(model.depth):
        wname ="weight"+str(k)
        convolutions.append((mod, wname))
    convolutions.append((model.msd.final_layer.linear, "weight"))
    return convolutions

def get_conv_masks_MSD(model):
    mod = model.msd.msd_block
    convolution_masks = []
    for k in range(model.depth):
        wname ="weight"+str(k)+"_mask"
        convolution_masks.append(getattr(mod, wname))
    convolution_masks.append(model.msd.final_layer.linear.weight_mask)
    return convolution_masks

def pruned_before_MSD(model):
    pruned_before = False
    mod = model.msd.msd_block
    for x in mod.named_buffers():
        if "mask" in x[0]:
            pruned_before = True
            break
    return pruned_before

def prune_biases_MSD(model):
    # final_layer biases are for output channels, will never be pruned
    conv_masks = get_conv_masks_MSD(model)
    mod = model.msd.msd_block
    model_device = conv_masks[0].device
    bias_mask = torch.Tensor(np.ones(mod.bias.size())).to(model_device)
    it = 0
    for mask in conv_masks:
        if it == model.depth: # We do not prune the output biases
            break
        if mask.sum() == 0:
            bias_mask[it] = 0
        it += 1
    method = prune.CustomFromMask(bias_mask)
    method.apply(mod, "bias", bias_mask)

def Prune_Redundant_Convolutions_MSD(model):
    conv_masks = get_conv_masks_MSD(model)
   
    # In this loop, we check for each output channel of each layer if all its input channels
    # are pruned. If so, that channel is pruned and afterwards the accompanying biases.
    it = 0
    count = 0
    for cmask in conv_masks:
        if it == 0 or it == model.depth:
            it += 1
            continue
        #Of size [c_out, c_in, ...]
        mask = conv_masks[it]
        for cin in range(it):
            prev_mask = conv_masks[cin]
            for i in range(prev_mask.size()[0]):
                if prev_mask[i].sum() == 0:
                    if mask[:,model.c_in + cin,:].sum() != 0:
                        count += 1
                    mask[:,model.c_in + cin,:] = 0
        it += 1
    # If all output convolutions in 1x1 are pruned, then that connected layer can be pruned.
    final_mask = conv_masks[-1]
    for cin in range(model.c_in, final_mask.size()[1]):
        # This is because c_in input channels are condensed into 1 by first layer
        prev_mask = conv_masks[cin - model.c_in]
        if prev_mask.sum() == 0:
            final_mask[:,cin] = 0
    #cmask = conv_masks[-1]
    #method = prune.CustomFromMask(cmask)
    #method.apply(model.msd.final_layer.linear, "weight", cmask)
    print("Pruned {} redundant convolutions.".format(count))
    prune_biases_MSD(model)

def get_convs_ResNet50(model):
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

def get_conv_masks_ResNet50(model):
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

def get_batchnorms_ResNet50(model):
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

def apply_mask_to_batchnorm_ResNet50(pmodel):
    conv_masks = get_conv_masks_ResNet50(pmodel)
    names = ['weight','bias']
    batchnorms_to_prune = get_batchnorms_ResNet50(pmodel)
    it = 0
    for modul in batchnorms_to_prune:
        if modul is None:
            it += 1
            continue
        mask = conv_masks[it]
        model_device = mask.device
        bn_mask = torch.Tensor(np.ones(mask.size()[0])).to(model_device)
        for i in range(mask.size()[0]):
            if mask[i].sum() == 0:
                bn_mask[i] = 0
        method = prune.CustomFromMask(bn_mask)
        for nam in names:
            method.apply(modul, nam, bn_mask)
        it += 1

def fraction_pruned_convs_ResNet50(model):
    r"""
    Return the fraction of pruned convs of ResNet50 model.
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
    r"""
    Return the fraction of pruned convs of MSD model.
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

def Prune_Redundant_Convolutions_ResNet50(pmodel, bn_thrs = 1e-10):
    parameters_to_prune = get_convs_ResNet50(pmodel)
    conv_masks = get_conv_masks_ResNet50(pmodel)
    batchnorms = get_batchnorms_ResNet50(pmodel)
   
    # In this loop, we check for each output channel of each layer if all its input channels
    # are pruned. If so, that channel is pruned and afterwards the accompanying batchnorm.
    it = 0
    count = 0
    for modul, nam in parameters_to_prune: #This is only to throw error if misaligned lists
        if it == 0:
            it += 1
            continue
        if it in [4,14,27,46]: # Downsample layer:
            prev_mask = conv_masks[it-4]
            mask = conv_masks[it]
            for i in range(prev_mask.size()[0]):
                if prev_mask[i].sum() == 0:
                    if mask[:,i].sum() != 0:
                        count += 1
                    mask[:,i] = 0
        elif it in [5,15,28,47]: # Layer after downsample layer:
            prev_mask1 = conv_masks[it-2]
            prev_mask2 = conv_masks[it-1]
            mask = conv_masks[it]
            for i in range(prev_mask1.size()[0]):
                if prev_mask1[i].sum() == 0 and prev_mask2[i].sum() == 0:
                    if mask[:,i].sum() != 0:
                        count += 1
                    mask[:,i] = 0
        else:
            prev_mask = conv_masks[it-1]
            mask = conv_masks[it]
            for i in range(prev_mask.size()[0]):
                if prev_mask[i].sum() == 0:
                    if mask[:,i].sum() != 0:
                        count += 1
                    mask[:,i] = 0
        it += 1
    apply_mask_to_batchnorm_ResNet50(pmodel)
    print("Pruned redundancies type 1:", count)

    # There are also nodes that are not pruned but always output zero_-valued images due to 
    # ReLU. These nodes can be found by finding running_variances that have gone to 0.
    it = 0
    for batnorm in batchnorms:
        if batnorm is None:
            it += 1
            continue
        bat_rvar = batnorm.running_var.data.cpu().detach().numpy()
        conv_layer_mask = conv_masks[it]
        conv_layer_mask[bat_rvar < bn_thrs] = 0
        it += 1
    apply_mask_to_batchnorm_ResNet50(pmodel)

def FourierSVD_LongPathMultiply_ResNet50(pmodel, tot_perc, batch=True, skip=True, Redun=True, ScaleDim=False):
    # perc refers to the total percentage of pruned convolutions we want to achieve
    if pruned_before_ResNet50(pmodel):
        prun_conv, tot_conv, frac_prun = fraction_pruned_convs_ResNet50(pmodel)
        if frac_prun >= 1 - tot_perc:
            print("No pruning to be done")
            return pmodel
        perc = 1.0 - tot_perc/(1.0 - frac_prun)
    else:
        perc = 1.0 - tot_perc
    print("Frac pruned convs should be:", 1 - tot_perc)

    M = 7 # Size of temporary input image
    # If the largest convolution is KxK, M=K is sufficient
    # A larger value will just increase each norm by a factor of (M/K)
    # However, a smaller M will give misleading results.
    order = 'max' # SVD-norm used for pruning
    parameters_to_prune = get_convs_ResNet50(pmodel)
    batchnorms = get_batchnorms_ResNet50(pmodel)

    # These are the indices of the downsample layer and where they point to
    downsample_idxs = [4,14,27,46]
    after_downsample_idxs = [5,15,28,47]
    # These are the other skip connections which are just identity mappings,
    #  when the identity points to a downsample layer, there are several connections
    skip_idxs = np.array([(8,3),(8,4),(11,7),(18,13),(18,14),(21,17),(24,20),(31,26),(31,27),(34,30),(37,33),(40,36),(43,39),(50,45),(50,46)])

    # If we are considering redundancy pruning, we must remove zero-images to avoid
    #  batch-scaling to skew the norms
    if pruned_before_ResNet50(pmodel) and Redun:
        Prune_Redundant_Convolutions_ResNet50(pmodel)

    nr_nodes = pmodel.c_in # start at c_in because of number of input channels
    for modul, nam in parameters_to_prune:
        orig = getattr(modul, nam)
        nr_nodes += orig.size()[0]
    print("Number of nodes in graph:", nr_nodes)

    norm_adjacency_matrix = np.zeros(shape=(nr_nodes,nr_nodes),dtype=np.float32)
    if skip:
        skip_connection_matrix = np.zeros(shape=(nr_nodes,nr_nodes),dtype=np.bool)
    else:
        skip_connection_matrix = None
    count_row = pmodel.c_in
    count_col = 0
    it = 0
    row_counts = []
    col_counts = []
    #max_batchmult = 0
    for modul, nam in parameters_to_prune:
        row_counts.append(count_row)
        col_counts.append(count_col)
        orig = getattr(modul, nam)
        norms = compute_FourierSVD_norms(orig, M, order).numpy()
        if ScaleDim:
            norms = norms / float(orig.size()[-1]*orig.size()[2])
        if skip: # Skip first so we can overwrite skip connections if necessary
            if it in skip_idxs[:,0]: # This layer has an incoming skip connection
                idx = np.where(skip_idxs[:,0] == it)[0][0]
                modul_in, nam_in = parameters_to_prune[skip_idxs[idx,1]]
                modul_between, nam_between = parameters_to_prune[skip_idxs[idx,1]+1]
                orig_in = getattr(modul_in, nam_in)
                orig_between = getattr(modul_between, nam_between)
                skip_size_row, skip_size_col = orig_in.size()[0], orig_in.size()[1]
                skip_row = count_row
                skip_col = count_col - skip_size_col - orig_between.size()[1]
                norm_adjacency_matrix[skip_row:skip_row+skip_size_row, skip_col:skip_col+skip_size_col] = 1
                skip_connection_matrix[skip_row:skip_row+skip_size_row, skip_col:skip_col+skip_size_col] = True
        if batchnorms[it] is not None:
            bn_mults = batchnorms[it].weight.data
            bn_rvar = batchnorms[it].running_var.data
            eps = batchnorms[it].eps
        for chan in range(norms.shape[0]):
            if batchnorms[it] is not None:
                batch_mult = np.abs(bn_mults[chan].cpu().numpy())
                batch_rvar = bn_rvar[chan].cpu().numpy()
                batch_mult = batch_mult / np.sqrt(batch_rvar + eps)
                #if batch_mult > max_batchmult:
                #    max_batchmult = batch_mult
            for c_in in range(norms.shape[1]):
                if it in downsample_idxs:
                    count_col_skip = col_counts[it-4]
                    if batch and batchnorms[it] is not None:
                        norm_adjacency_matrix[count_row+chan,count_col_skip+c_in] = norms[chan,c_in] * batch_mult
                    else:
                        norm_adjacency_matrix[count_row+chan,count_col_skip+c_in] = norms[chan,c_in]
                elif it in after_downsample_idxs:
                    count_col_skip = col_counts[it-1]
                    if batch and batchnorms[it] is not None:
                        norm_adjacency_matrix[count_row+chan,count_col+c_in] = norms[chan,c_in] * batch_mult
                        norm_adjacency_matrix[count_row+chan,count_col_skip+c_in] = norms[chan,c_in] * batch_mult
                    else:
                        norm_adjacency_matrix[count_row+chan,count_col+c_in] = norms[chan,c_in]
                        norm_adjacency_matrix[count_row+chan,count_col_skip+c_in] = norms[chan,c_in]
                else:
                    if batch and batchnorms[it] is not None:
                        norm_adjacency_matrix[count_row+chan,count_col+c_in] = norms[chan,c_in] * batch_mult
                    else:
                        norm_adjacency_matrix[count_row+chan,count_col+c_in] = norms[chan,c_in]
        #print(it, norm_adjacency_matrix.max())
        #print(np.where(norm_adjacency_matrix == np.amax(norm_adjacency_matrix)))
        count_row += norms.shape[0]
        count_col += norms.shape[1]
        it += 1
    #print(max_batchmult)
    #print(norm_adjacency_matrix.max())
    #print(np.where(norm_adjacency_matrix == np.amax(norm_adjacency_matrix)))
    #print(norm_adjacency_matrix[norm_adjacency_matrix != 0].size)
    print(norm_adjacency_matrix[norm_adjacency_matrix > 1].size)
    #get_norm_histogram(norm_adjacency_matrix)
    #draw_adjacency_matrix(norm_adjacency_matrix)

    pruned = gru.longest_path_prune(norm_adjacency_matrix, perc, pmodel.c_in, skip_connection_matrix)
    count_row = pmodel.c_in
    count_col = 0
    for modul, nam in parameters_to_prune:
        orig = getattr(modul, nam)
        mask = torch.ones_like(orig)
        default_mask = get_default_mask(modul, nam)
        for i in range(mask.size()[0]):
            for j in range(mask.size()[1]):
                if pruned[count_row + i,count_col + j]:
                    mask[i][j] = torch.zeros_like(orig[i][j])
        count_row += mask.size()[0]
        count_col += mask.size()[1]
        mask *= default_mask.to(dtype=mask.dtype)
        method = prune.CustomFromMask(mask)
        method.apply(modul, nam, mask)

    if Redun:
        Prune_Redundant_Convolutions_ResNet50(pmodel)
    else:
        apply_mask_to_batchnorm_ResNet50(pmodel)
    return pmodel

def Ln_Layer_ResNet50(pmodel, tot_perc, N=1, Redun=True):
    if pruned_before_ResNet50(pmodel):
        prun_conv, tot_conv, frac_prun = fraction_pruned_convs_ResNet50(pmodel)
        if frac_prun >= 1 - tot_perc:
            print("No pruning to be done")
            return pmodel
        perc = 1.0 - tot_perc/(1.0 - frac_prun)
    else:
        perc = 1.0 - tot_perc
    print("Frac pruned convs should be:", 1 - tot_perc)
    raise Exception("Not fit to use yet")

    parameters_to_prune = get_convs_ResNet50(pmodel)

    if pruned_before_ResNet50(pmodel) and Redun:
        Prune_Redundant_Convolutions_ResNet50(pmodel)

    layer_norms = []
    for modul, nam in parameters_to_prune:
        orig = getattr(modul, nam)
        norms = compute_Ln_norms_conv(orig, N).numpy()
        layer_norms.append(norms.tolist())

    conv_masks = []
    it = 0
    for modul, nam in parameters_to_prune:
        orig = getattr(modul, nam)
        lnorms = layer_norms[it]
        threshold = np.percentile(lnorms, 100*perc)
        mask = torch.ones_like(orig)
        default_mask = get_default_mask(modul, nam)
        for i in range(len(lnorms)):
            for j in range(len(lnorms[i])):
                if lnorms[i][j] < threshold:
                    mask[i][j] = torch.zeros_like(orig[i][j])
        mask *= default_mask.to(dtype=mask.dtype)
        it += 1
        method = prune.CustomFromMask(mask)
        method.apply(modul, nam, mask)

    if Redun:
        Prune_Redundant_Convolutions_ResNet50(pmodel)
    else:
        apply_mask_to_batchnorm_ResNet50(pmodel)
    return pmodel

def Ln_Global_ResNet50(pmodel, tot_perc, N=1, Redun=True, ScaleDim=False):
    if pruned_before_ResNet50(pmodel):
        prun_conv, tot_conv, frac_prun = fraction_pruned_convs_ResNet50(pmodel)
        if frac_prun >= 1 - tot_perc:
            print("No pruning to be done")
            return pmodel
        perc = 1.0 - tot_perc/(1.0 - frac_prun)
    else:
        perc = 1.0 - tot_perc
    print("Frac pruned convs should be:", 1 - tot_perc)

    parameters_to_prune = get_convs_ResNet50(pmodel)
    if pruned_before_ResNet50(pmodel) and Redun:
        Prune_Redundant_Convolutions_ResNet50(pmodel)

    all_norms = np.array([], dtype=np.float32)
    layer_norms = []
    for modul, nam in parameters_to_prune:
        orig = getattr(modul, nam)
        norms = compute_Ln_norms_conv(orig, N).numpy()
        if ScaleDim:
            norms = norms / float(orig.size()[-1]*orig.size()[2])
        all_norms = np.concatenate((all_norms, norms[norms > 0].flatten()))
        layer_norms.append(norms.tolist())
    threshold = np.percentile(all_norms, 100*perc)

    conv_masks = []
    it = 0
    for modul, nam in parameters_to_prune:
        orig = getattr(modul, nam)
        lnorms = layer_norms[it]
        mask = torch.ones_like(orig)
        default_mask = get_default_mask(modul, nam)
        for i in range(len(lnorms)):
            for j in range(len(lnorms[i])):
                if lnorms[i][j] < threshold:
                    mask[i][j] = torch.zeros_like(orig[i][j])
        mask *= default_mask.to(dtype=mask.dtype)
        it += 1
        method = prune.CustomFromMask(mask)
        method.apply(modul, nam, mask)

    if Redun:
        Prune_Redundant_Convolutions_ResNet50(pmodel)
    else:
        apply_mask_to_batchnorm_ResNet50(pmodel)
    return pmodel


def FourierSVD_Global_ResNet50(pmodel, tot_perc, batch=True, Redun=True):
    if pruned_before_ResNet50(pmodel):
        prun_conv, tot_conv, frac_prun = fraction_pruned_convs_ResNet50(pmodel)
        if frac_prun >= 1 - tot_perc:
            print("No pruning to be done")
            return pmodel
        perc = 1.0 - tot_perc/(1.0 - frac_prun)
    else:
        perc = 1.0 - tot_perc
    #perc = 1 - tot_perc
    print("Frac pruned convs should be:", 1 - tot_perc)

    M = 7 # Size of temporary input image
    order = 'max' # SVD-norm used for pruning
    parameters_to_prune = get_convs_ResNet50(pmodel)
    batchnorms = get_batchnorms_ResNet50(pmodel)

    if pruned_before_ResNet50(pmodel) and Redun:
        Prune_Redundant_Convolutions_ResNet50(pmodel)

    it = 0
    all_norms = np.array([], dtype=np.float32)
    layer_norms = []
    for modul, nam in parameters_to_prune:
        orig = getattr(modul, nam)
        norms = compute_FourierSVD_norms(orig, M, order).numpy()
        if batch and batchnorms[it] is not None:
            bn_mults = batchnorms[it].weight.data
            bn_rvar = batchnorms[it].running_var.data
            eps = batchnorms[it].eps
            for chan in range(norms.shape[0]):
                batch_mult = np.abs(bn_mults[chan].cpu().numpy())
                batch_rvar = bn_rvar[chan].cpu().numpy()
                batch_mult = batch_mult / np.sqrt(batch_rvar + eps)
                norms[chan,:] *= batch_mult
        #all_norms = np.concatenate((all_norms, norms.flatten()))
        all_norms = np.concatenate((all_norms, norms[norms > 0].flatten()))
        layer_norms.append(norms.tolist())
        it += 1
    threshold = np.percentile(all_norms, 100*perc)

    masks = []
    it = 0
    count_masked = 0
    count_notmasked = 0
    for modul, nam in parameters_to_prune:
        orig = getattr(modul, nam)
        lnorms = layer_norms[it]
        mask = torch.ones_like(orig)
        default_mask = get_default_mask(modul, nam)
        for i in range(len(lnorms)):
            for j in range(len(lnorms[i])):
                if lnorms[i][j] <= threshold:
                    count_masked += 1
                    mask[i][j] = torch.zeros_like(orig[i][j])
                else:
                    count_notmasked += 1
        mask *= default_mask.to(dtype=mask.dtype)
        it += 1
        method = prune.CustomFromMask(mask)
        method.apply(modul, nam, mask)
    #print(count_masked, count_notmasked)
    #print(perc, fraction_pruned_convs_ResNet50(pmodel))
    if Redun:
        Prune_Redundant_Convolutions_ResNet50(pmodel)
    else:
        apply_mask_to_batchnorm_ResNet50(pmodel)
    print("Redun is", Redun, perc, fraction_pruned_convs_ResNet50(pmodel))
    return pmodel

def Random_Global_ResNet50(pmodel, tot_perc):
    if pruned_before_ResNet50(pmodel):
        prun_conv, tot_conv, frac_prun = fraction_pruned_convs_ResNet50(pmodel)
        if frac_prun >= 1 - tot_perc:
            print("No pruning to be done")
            return pmodel
    raise Exception("Test which one to use first")
    #    perc = 1.0 - tot_perc/(1.0 - frac_prun)
    #else:
    #    perc = 1.0 - tot_perc
    perc = 1 - tot_perc
    print("Frac pruned convs should be:", 1 - tot_perc)

    parameters_to_prune = get_convs_ResNet50(pmodel)

    all_norms = np.array([], dtype=np.float32)
    layer_norms = []
    for modul, nam in parameters_to_prune:
        orig = getattr(modul, nam)
        random_norms = torch.rand_like(orig[:,:,0,0]).cpu().numpy() # Between [0,1)
        random_norms += 0.0001 # We don't want 0 norms because it messes up iterative pruning
        all_norms = np.concatenate((all_norms, random_norms[random_norms > 0].flatten()))
        layer_norms.append(random_norms.tolist())
    threshold = np.percentile(all_norms, 100*perc)

    masks = []
    it = 0
    tot_prms = 0
    tot_ones = 0
    for modul, nam in parameters_to_prune:
        orig = getattr(modul, nam)
        lnorms = layer_norms[it]
        mask = torch.ones_like(orig)
        default_mask = get_default_mask(modul, nam)
        for i in range(len(lnorms)):
            for j in range(len(lnorms[i])):
                if lnorms[i][j] < threshold:
                    mask[i][j] = torch.zeros_like(orig[i][j])
        mask *= default_mask.to(dtype=mask.dtype)
        it += 1
        tot_prms += mask.numel()
        tot_ones += mask.sum()
        method = prune.CustomFromMask(mask)
        method.apply(modul, nam, mask)
    apply_mask_to_batchnorm_ResNet50(pmodel)
    return pmodel

def FourierSVD_LongPathMultiply_MSD(pmodel, tot_perc, Redun=True, ScaleDim=False, verbose=True):
    r"""Prunes a percentage of the weights in the entire MSD network at once, not only per layer,
    using a longest path calculated by log-multiplication distance on some Fourier-SVD norm.
    """
    if pruned_before_MSD(pmodel):
        prun_conv, tot_conv, frac_prun = fraction_pruned_convs_MSD(pmodel)
        if frac_prun >= 1 - tot_perc:
            if verbose:
                print("No pruning to be done")
            return pmodel
        perc = 1.0 - tot_perc/(1.0 - frac_prun)
    else:
        perc = 1.0 - tot_perc

    if pmodel.width > 1:
        raise Exception("Not implemented for MS-D network with width > 1!")
    M = 64 # Size of temporary input image, (larger because of dilations)
    order = 'max' # SVD-norm used for pruning
    parameters_to_prune = get_convs_MSD(pmodel)

    if pruned_before_MSD(pmodel) and Redun:
        Prune_Redundant_Convolutions_MSD(pmodel)

    arr_size = pmodel.depth + pmodel.c_in + pmodel.c_out
    norm_adjacency_matrix = np.zeros(shape=(arr_size, arr_size), dtype=np.float32)
    count = pmodel.c_in
    for modul, nam in parameters_to_prune:
        orig = getattr(modul, nam)
        if orig.size()[-1] == 1:
            orig1x1 = orig.unsqueeze(2)
            norms = compute_FourierSVD_norms(orig1x1, M, order).numpy()
            norms = norms*1.0#TODO
            for i in range(norms.shape[0]):
                for k in range(norms.shape[1]):
                    norm_adjacency_matrix[count+i,k] = norms[i,k]
        else:
            norms = compute_FourierSVD_norms(orig, M, order).numpy()[0]
            norms = norms*1.0#TODO
            if ScaleDim:
                norms = norms / float(orig.size()[-1]*orig.size()[2]) # Scale with conv size
            for k in range(norms.shape[0]):
                norm_adjacency_matrix[count,k] = norms[k]
        count += 1

    pruned = gru.longest_path_prune(norm_adjacency_matrix, perc, pmodel.c_in)
    it = pmodel.c_in
    for modul, nam in parameters_to_prune:
        orig = getattr(modul, nam)
        mask = torch.ones_like(orig)
        default_mask = get_default_mask(modul, nam)
        for i in range(mask.size()[0]):
            for j in range(mask.size()[1]):
                if pruned[it+i,j]:
                    mask[i][j] = torch.zeros_like(orig[i][j])
        it += 1
        mask *= default_mask.to(dtype=mask.dtype)
        method = prune.CustomFromMask(mask)
        method.apply(modul, nam, mask)

    if Redun:
        Prune_Redundant_Convolutions_MSD(pmodel)
    else:
        prune_biases_MSD(pmodel)
    return pmodel

def FourierSVD_Global_MSD(pmodel, tot_perc, Redun=True, ScaleDim=False, verbose=True):
    if pruned_before_MSD(pmodel):
        prun_conv, tot_conv, frac_prun = fraction_pruned_convs_MSD(pmodel)
        if frac_prun >= 1 - tot_perc:
            if verbose:
                print("No pruning to be done")
            return pmodel
        perc = 1.0 - tot_perc/(1.0 - frac_prun)
    else:
        perc = 1.0 - tot_perc
    
    M = 64 # Size of temporary input image, (larger because of dilations)
    order = 'max' # SVD-norm used for pruning
    parameters_to_prune = get_convs_MSD(pmodel)

    if pruned_before_MSD(pmodel) and Redun:
        Prune_Redundant_Convolutions_MSD(pmodel)

    all_norms = np.array([], dtype=np.float32)
    layer_norms = []
    for modul, nam in parameters_to_prune:
        orig = getattr(modul, nam)
        if orig.size()[-1] == 1:
            orig1x1 = orig.unsqueeze(2)
            norms = compute_FourierSVD_norms(orig1x1, M, order).numpy()
        else:
            norms = compute_FourierSVD_norms(orig, M, order).numpy()
            if ScaleDim:
                norms = norms / float(orig.size()[-1]*orig.size()[2]) # Scale with conv size
        all_norms = np.concatenate((all_norms, norms[norms > 0].flatten()))
        layer_norms.append(norms.tolist())
    threshold = np.percentile(all_norms, 100*perc)

    it = 0
    for modul, nam in parameters_to_prune:
        orig = getattr(modul, nam)
        lnorms = layer_norms[it]
        mask = torch.ones_like(orig)
        default_mask = get_default_mask(modul, nam)
        for i in range(len(lnorms)):
            for j in range(len(lnorms[i])):
                if lnorms[i][j] < threshold:
                    mask[i][j] = torch.zeros_like(orig[i][j])
        mask *= default_mask.to(dtype=mask.dtype)
        it += 1
        method = prune.CustomFromMask(mask)
        method.apply(modul, nam, mask)

    if Redun:
        Prune_Redundant_Convolutions_MSD(pmodel)
    else:
        prune_biases_MSD(pmodel)
    return pmodel

def Ln_Global_MSD(pmodel, tot_perc, Redun=True, ScaleDim=False, verbose=True):
    N = 1 # Order of Ln norm
    if pruned_before_MSD(pmodel):
        prun_conv, tot_conv, frac_prun = fraction_pruned_convs_MSD(pmodel)
        if frac_prun >= 1 - tot_perc:
            if verbose:
                print("No pruning to be done")
            return pmodel
        perc = 1.0 - tot_perc/(1.0 - frac_prun)
    else:
        perc = 1.0 - tot_perc
    parameters_to_prune = get_convs_MSD(pmodel)

    if pruned_before_MSD(pmodel) and Redun:
        Prune_Redundant_Convolutions_MSD(pmodel)

    all_norms = np.array([], dtype=np.float32)
    layer_norms = []
    for modul, nam in parameters_to_prune:
        orig = getattr(modul, nam)
        if orig.size()[-1] == 1:
            orig1x1 = orig.unsqueeze(2)
            norms = compute_Ln_norms_conv(orig1x1, N).numpy()
        else:
            norms = compute_Ln_norms_conv(orig, N).numpy()
            if ScaleDim:
                norms = norms / float(orig.size()[-1]*orig.size()[2]) # Scale with conv size
        all_norms = np.concatenate((all_norms, norms[norms > 0].flatten()))
        layer_norms.append(norms.tolist())
    threshold = np.percentile(all_norms, 100*perc)

    it = 0
    for modul, nam in parameters_to_prune:
        orig = getattr(modul, nam)
        lnorms = layer_norms[it]
        mask = torch.ones_like(orig)
        default_mask = get_default_mask(modul, nam)
        for i in range(len(lnorms)):
            for j in range(len(lnorms[i])):
                if lnorms[i][j] < threshold:
                    mask[i][j] = torch.zeros_like(orig[i][j])
        mask *= default_mask.to(dtype=mask.dtype)
        it += 1
        method = prune.CustomFromMask(mask)
        method.apply(modul, nam, mask)

    if Redun:
        Prune_Redundant_Convolutions_MSD(pmodel)
    else:
        prune_biases_MSD(pmodel)
    return pmodel

def Random_Global_MSD(pmodel, tot_perc):
    if pruned_before_MSD(pmodel):
        prun_conv, tot_conv, frac_prun = fraction_pruned_convs_MSD(pmodel)
        if frac_prun >= 1 - tot_perc:
            print("No pruning to be done")
            return pmodel
        perc = 1.0 - tot_perc/(1.0 - frac_prun)
    else:
        perc = 1.0 - tot_perc
    parameters_to_prune = get_convs_MSD(pmodel)

    all_norms = np.array([], dtype=np.float32)
    layer_norms = []
    for modul, nam in parameters_to_prune:
        orig = getattr(modul, nam)
        if orig.size()[-1] == 1:
            random_norms = torch.rand_like(orig[:,:,0]).cpu().numpy() # Between [0,1)
        else:
            random_norms = torch.rand_like(orig[:,:,0,0]).cpu().numpy() # Between [0,1)
        random_norms += 0.0001 # We don't want 0 norms because it messes up iterative pruning
        all_norms = np.concatenate((all_norms, random_norms[random_norms > 0].flatten()))
        layer_norms.append(random_norms.tolist())
    threshold = np.percentile(all_norms, 100*perc)

    it = 0
    for modul, nam in parameters_to_prune:
        orig = getattr(modul, nam)
        lnorms = layer_norms[it]
        mask = torch.ones_like(orig)
        default_mask = get_default_mask(modul, nam)
        for i in range(len(lnorms)):
            for j in range(len(lnorms[i])):
                if lnorms[i][j] < threshold:
                    mask[i][j] = torch.zeros_like(orig[i][j])
        mask *= default_mask.to(dtype=mask.dtype)
        it += 1
        method = prune.CustomFromMask(mask)
        method.apply(modul, nam, mask)
    prune_biases_MSD(pmodel)
    return pmodel

def get_conv_masks_MSD3x3(model):
    mod = model.msd.msd_block
    convolution_masks = []
    for k in range(model.depth):
        wname ="weight"+str(k)+"_mask"
        convolution_masks.append(getattr(mod, wname))
    return convolution_masks

def prune_biases_MSD3x3(model):
    conv_masks = get_conv_masks_MSD3x3(model)
    mod = model.msd.msd_block
    model_device = conv_masks[0].device
    bias_mask = torch.Tensor(np.ones(mod.bias.size())).to(model_device)
    it = 0
    for mask in conv_masks:
        if mask.sum() == 0:
            bias_mask[it] = 0
        it += 1
    method = prune.CustomFromMask(bias_mask)
    method.apply(mod, "bias", bias_mask)

def Prune_Redundant_Convolutions_MSD3x3(model):
    conv_masks = get_conv_masks_MSD3x3(model)
   
    # In this loop, we check for each output channel of each layer if all its input channels
    # are pruned. If so, that channel is pruned and afterwards the accompanying biases.
    it = 0
    count = 0
    for cmask in conv_masks:
        if it == 0 or it == model.depth:
            it += 1
            continue
        #Of size [c_out, c_in, ...]
        mask = conv_masks[it]
        for cin in range(it):
            prev_mask = conv_masks[cin]
            for i in range(prev_mask.size()[0]):
                if prev_mask[i].sum() == 0:
                    if mask[:,model.c_in + cin,:].sum() != 0:
                        count += 1
                    mask[:,model.c_in + cin,:] = 0
        it += 1
    print("Pruned {} redundant convolutions.".format(count))
    prune_biases_MSD3x3(model)

def fraction_pruned_convs_MSD3x3(model):
    r"""
    Return the fraction of pruned convs of MSD model.
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

def Random_Global_MSD_3x3(pmodel, tot_perc):
    if pruned_before_MSD(pmodel):
        prun_conv, tot_conv, frac_prun = fraction_pruned_convs_MSD3x3(pmodel)
        if frac_prun >= 1 - tot_perc:
            print("No pruning to be done")
            return pmodel
        perc = 1.0 - tot_perc/(1.0 - frac_prun)
    else:
        perc = 1.0 - tot_perc
    parameters_to_prune = get_convs_MSD(pmodel)
    parameters_to_prune = parameters_to_prune[:-1] # NOTE: Different

    all_norms = np.array([], dtype=np.float32)
    layer_norms = []
    for modul, nam in parameters_to_prune:
        orig = getattr(modul, nam)
        if orig.size()[-1] == 1:
            random_norms = torch.rand_like(orig[:,:,0]).cpu().numpy() # Between [0,1)
        else:
            random_norms = torch.rand_like(orig[:,:,0,0]).cpu().numpy() # Between [0,1)
        random_norms += 0.0001 # We don't want 0 norms because it messes up iterative pruning
        all_norms = np.concatenate((all_norms, random_norms[random_norms > 0].flatten()))
        layer_norms.append(random_norms.tolist())
    threshold = np.percentile(all_norms, 100*perc)

    it = 0
    for modul, nam in parameters_to_prune:
        orig = getattr(modul, nam)
        lnorms = layer_norms[it]
        mask = torch.ones_like(orig)
        default_mask = get_default_mask(modul, nam)
        for i in range(len(lnorms)):
            for j in range(len(lnorms[i])):
                if lnorms[i][j] < threshold:
                    mask[i][j] = torch.zeros_like(orig[i][j])
        mask *= default_mask.to(dtype=mask.dtype)
        it += 1
        method = prune.CustomFromMask(mask)
        method.apply(modul, nam, mask)
    #prune_biases_MSD(pmodel) #NOTE: Different
    return pmodel

def Ln_Layer_MSD_3x3(pmodel, tot_perc, Redun=True):
    N = 1 # Order of Ln norm
    ispruned = pruned_before_MSD(pmodel)
    if pruned_before_MSD(pmodel):
        prun_conv, tot_conv, frac_prun = fraction_pruned_convs_MSD3x3(pmodel)#NOTE
        if frac_prun >= 1 - tot_perc:
            print("No pruning to be done")
            return pmodel
        perc = 1.0 - tot_perc/(1.0 - frac_prun)
    else:
        perc = 1.0 - tot_perc
    parameters_to_prune = get_convs_MSD(pmodel)
    parameters_to_prune = parameters_to_prune[:-1] # NOTE: Different
    if ispruned:
        conv_masks = get_conv_masks_MSD3x3(pmodel)

    if pruned_before_MSD(pmodel) and Redun:
        Prune_Redundant_Convolutions_MSD3x3(pmodel)#NOTE

    dm = 1
    for k in range(pmodel.depth):
        wname ="weight"+str(k)
        customln_structured(pmodel.msd.msd_block, name=wname, amount=perc, n=N, dim=dm)

    """
    layer_norms = []
    for modul, nam in parameters_to_prune:
        orig = getattr(modul, nam)
        if orig.size()[-1] == 1:
            raise Exception("Something went wrong")#NOTE
        else:
            norms = compute_Ln_norms_conv(orig, N).numpy()
        layer_norms.append(norms.tolist())

    it = 0
    eps = 0.0001
    for modul, nam in parameters_to_prune:
        orig = getattr(modul, nam)
        lnorms = layer_norms[it]

        # get perc for every layer seperately
        lperc = 0.0
        current_frac = 0.0
        if ispruned:
            old_mask = conv_masks[it]
            count = old_mask.sum(axis=-1)
            count = count.sum(axis=-1)
            pruned_convs = (count == 0).sum().item()
            tot_convs = len(lnorms) * len(lnorms[0])
            current_frac = pruned_convs / float(tot_convs)
        if current_frac < 1 - tot_perc:
            lperc = 1.0 - tot_perc/(1.0 - current_frac)

        # Because of how np.percentile works, we need to fix threshold boundaries
        minx = np.min(np.array(lnorms)) - eps
        maxx = np.max(np.array(lnorms)) + eps
        lnormstemp = np.array(lnorms).flatten()
        lnormstemp = np.concatenate((lnormstemp, np.array([minx,maxx])))
        threshold = np.percentile(lnormstemp, 100*(1.0 - tot_perc), interpolation='lower')

        mask = torch.ones_like(orig)
        default_mask = get_default_mask(modul, nam)
        for i in range(len(lnorms)):
            for j in range(len(lnorms[i])):
                if lnorms[i][j] < threshold:
                    mask[i][j] = torch.zeros_like(orig[i][j])
        mask *= default_mask.to(dtype=mask.dtype)
        it += 1
        method = prune.CustomFromMask(mask)
        method.apply(modul, nam, mask)
    """

    if Redun:
        Prune_Redundant_Convolutions_MSD3x3(pmodel)#NOTE
    else:
        prune_biases_MSD3x3(pmodel)#NOTE
    return pmodel

def Ln_Global_MSD_3x3(pmodel, tot_perc, Redun=True, ScaleDim=False, verbose=True):
    N = 1 # Order of Ln norm
    if pruned_before_MSD(pmodel):
        prun_conv, tot_conv, frac_prun = fraction_pruned_convs_MSD3x3(pmodel)#NOTE
        if frac_prun >= 1 - tot_perc:
            if verbose:
                print("No pruning to be done")
            return pmodel
        perc = 1.0 - tot_perc/(1.0 - frac_prun)
    else:
        perc = 1.0 - tot_perc
    parameters_to_prune = get_convs_MSD(pmodel)
    parameters_to_prune = parameters_to_prune[:-1] # NOTE: Different

    if pruned_before_MSD(pmodel) and Redun:
        Prune_Redundant_Convolutions_MSD3x3(pmodel)#NOTE

    all_norms = np.array([], dtype=np.float32)
    layer_norms = []
    for modul, nam in parameters_to_prune:
        orig = getattr(modul, nam)
        if orig.size()[-1] == 1:
            raise Exception("Something went wrong")#NOTE
            orig1x1 = orig.unsqueeze(2)
            norms = compute_Ln_norms_conv(orig1x1, N).numpy()
        else:
            norms = compute_Ln_norms_conv(orig, N).numpy()
            if ScaleDim:
                norms = norms / float(orig.size()[-1]*orig.size()[2]) # Scale with conv size
        all_norms = np.concatenate((all_norms, norms[norms > 0].flatten()))
        layer_norms.append(norms.tolist())
    threshold = np.percentile(all_norms, 100*perc)

    it = 0
    for modul, nam in parameters_to_prune:
        orig = getattr(modul, nam)
        lnorms = layer_norms[it]
        mask = torch.ones_like(orig)
        default_mask = get_default_mask(modul, nam)
        for i in range(len(lnorms)):
            for j in range(len(lnorms[i])):
                if lnorms[i][j] < threshold:
                    mask[i][j] = torch.zeros_like(orig[i][j])
        mask *= default_mask.to(dtype=mask.dtype)
        it += 1
        method = prune.CustomFromMask(mask)
        method.apply(modul, nam, mask)

    if Redun:
        Prune_Redundant_Convolutions_MSD3x3(pmodel)#NOTE
    else:
        prune_biases_MSD3x3(pmodel)#NOTE
    return pmodel

def FourierSVD_LongPathMultiply_MSD_3x3(pmodel, tot_perc, Redun=True, ScaleDim=False, verbose=True):
    r"""Prunes a percentage of the weights in the entire MSD network at once, not only per layer,
    using a longest path calculated by log-multiplication distance on some Fourier-SVD norm.
    """
    if pruned_before_MSD(pmodel):
        prun_conv, tot_conv, frac_prun = fraction_pruned_convs_MSD3x3(pmodel)#NOTE
        if frac_prun >= 1 - tot_perc:
            if verbose:
                print("No pruning to be done")
            return pmodel
        perc = 1.0 - tot_perc/(1.0 - frac_prun)
    else:
        perc = 1.0 - tot_perc

    if pmodel.width > 1:
        raise Exception("Not implemented for MS-D network with width > 1!")
    M = 64 # Size of temporary input image, (larger because of dilations)
    order = 'max' # SVD-norm used for pruning
    parameters_to_prune = get_convs_MSD(pmodel)
    parameters_to_prune = parameters_to_prune[:-1] # NOTE: Different

    if pruned_before_MSD(pmodel) and Redun:
        Prune_Redundant_Convolutions_MSD3x3(pmodel)#NOTE

    arr_size = pmodel.depth + pmodel.c_in + pmodel.c_out
    norm_adjacency_matrix = np.zeros(shape=(arr_size, arr_size), dtype=np.float32)
    count = pmodel.c_in
    for modul, nam in parameters_to_prune:
        orig = getattr(modul, nam)
        if orig.size()[-1] == 1:
            raise Exception("Something went wrong")#NOTE
            orig1x1 = orig.unsqueeze(2)
            norms = compute_FourierSVD_norms(orig1x1, M, order).numpy()
            for i in range(norms.shape[0]):
                for k in range(norms.shape[1]):
                    norm_adjacency_matrix[count+i,k] = norms[i,k]
        else:
            norms = compute_FourierSVD_norms(orig, M, order).numpy()[0]
            if ScaleDim:
                norms = norms / float(orig.size()[-1]*orig.size()[2]) # Scale with conv size
            for k in range(norms.shape[0]):
                norm_adjacency_matrix[count,k] = norms[k]
        count += 1

    pruned = gru.longest_path_prune(norm_adjacency_matrix, perc, pmodel.c_in)
    it = pmodel.c_in
    for modul, nam in parameters_to_prune:
        orig = getattr(modul, nam)
        mask = torch.ones_like(orig)
        default_mask = get_default_mask(modul, nam)
        for i in range(mask.size()[0]):
            for j in range(mask.size()[1]):
                if pruned[it+i,j]:
                    mask[i][j] = torch.zeros_like(orig[i][j])
        it += 1
        mask *= default_mask.to(dtype=mask.dtype)
        method = prune.CustomFromMask(mask)
        method.apply(modul, nam, mask)

    if Redun:
        Prune_Redundant_Convolutions_MSD3x3(pmodel)#NOTE
    else:
        prune_biases_MSD3x3(pmodel)#NOTE
    return pmodel

def FourierSVD_Global_MSD_3x3(pmodel, tot_perc, Redun=True, ScaleDim=False, verbose=True):
    if pruned_before_MSD(pmodel):
        prun_conv, tot_conv, frac_prun = fraction_pruned_convs_MSD3x3(pmodel)#NOTE
        if frac_prun >= 1 - tot_perc:
            if verbose:
                print("No pruning to be done")
            return pmodel
        perc = 1.0 - tot_perc/(1.0 - frac_prun)
    else:
        perc = 1.0 - tot_perc

    M = 64 # Size of temporary input image, (larger because of dilations)
    order = 'max' # SVD-norm used for pruning
    parameters_to_prune = get_convs_MSD(pmodel)
    parameters_to_prune = parameters_to_prune[:-1] # NOTE: Different

    if pruned_before_MSD(pmodel) and Redun:
        Prune_Redundant_Convolutions_MSD3x3(pmodel)#NOTE

    all_norms = np.array([], dtype=np.float32)
    layer_norms = []
    for modul, nam in parameters_to_prune:
        orig = getattr(modul, nam)
        if orig.size()[-1] == 1:
            raise Exception("Something went wrong")#NOTE
            orig1x1 = orig.unsqueeze(2)
            norms = compute_FourierSVD_norms(orig1x1, M, order).numpy()
        else:
            norms = compute_FourierSVD_norms(orig, M, order).numpy()
            if ScaleDim:
                norms = norms / float(orig.size()[-1]*orig.size()[2]) # Scale with conv size
        all_norms = np.concatenate((all_norms, norms[norms > 0].flatten()))
        layer_norms.append(norms.tolist())
    threshold = np.percentile(all_norms, 100*perc)
    
    it = 0
    for modul, nam in parameters_to_prune:
        orig = getattr(modul, nam)
        lnorms = layer_norms[it]
        mask = torch.ones_like(orig)
        default_mask = get_default_mask(modul, nam)
        for i in range(len(lnorms)):
            for j in range(len(lnorms[i])):
                if lnorms[i][j] < threshold:
                    mask[i][j] = torch.zeros_like(orig[i][j])
        mask *= default_mask.to(dtype=mask.dtype)
        it += 1
        method = prune.CustomFromMask(mask)
        method.apply(modul, nam, mask)

    if Redun:
        Prune_Redundant_Convolutions_MSD3x3(pmodel)#NOTE
    else:
        prune_biases_MSD3x3(pmodel)#NOTE
    return pmodel

import torch
import msd_pytorch as mp
import numpy as np
import torch.nn as nn
import torch.nn.utils.prune as prune

from torch_pruning_methods import *
from pruning_utils import *
import graph_algorithms as gru


def LEAN_SV_ResNet50(pmodel, tot_perc, Redun=True):
    r"""Prune FCN-ResNet50 model using LEAN pruning.

    Args:
        - pmodel: PyTorch-model of FCN-ResNet50 to be pruned.
        - tot_perc (float): The total fraction of convolutions
            we want pruned at the end of this pruning step.
        - Redun (bool): Whether to perform redundancy pruning
            at the end of the pruning phase. Default is True.
    """
    # Calculate by what percentage the model needs to be pruned
    # to obtain 'tot_perc' percent of pruning. It can e.g. require
    # less than the expected percentage because redundancy pruning
    # removed a significant amount in the previous pruning phase.
    if pruned_before_ResNet50(pmodel):
        prun_conv, tot_conv, frac_prun = fraction_pruned_convs_ResNet50(pmodel)
        if frac_prun >= 1 - tot_perc:
            print("No pruning to be done")
            return pmodel
        perc = 1.0 - tot_perc/(1.0 - frac_prun)
    else:
        perc = 1.0 - tot_perc

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

    if pruned_before_ResNet50(pmodel) and Redun:
        Prune_Redundant_Convolutions_ResNet50(pmodel)

    nr_nodes = pmodel.c_in # start at c_in because of number of input channels
    for modul, nam in parameters_to_prune:
        orig = getattr(modul, nam)
        nr_nodes += orig.size()[0]
    print("Number of nodes in graph:", nr_nodes)

    norm_adjacency_matrix = np.zeros(shape=(nr_nodes,nr_nodes),dtype=np.float32)
    skip_connection_matrix = np.zeros(shape=(nr_nodes,nr_nodes),dtype=np.bool)

    ### Create the norm graph
    count_row = pmodel.c_in
    count_col = 0
    it = 0
    for modul, nam in parameters_to_prune:
        orig = getattr(modul, nam)

        ### Compute the convolution norms
        norms = compute_FourierSVD_norms(orig, M, order).numpy()

        ### Include the skip connections
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

        # Incoporate batch normalization scaling and set the edge values
        # in the adjacency matrix.
        if batchnorms[it] is not None:
            bn_mults = batchnorms[it].weight.data
            bn_rvar = batchnorms[it].running_var.data
            eps = batchnorms[it].eps
        for chan in range(norms.shape[0]):
            if batchnorms[it] is not None:
                batch_mult = np.abs(bn_mults[chan].cpu().numpy())
                batch_rvar = bn_rvar[chan].cpu().numpy()
                batch_mult = batch_mult / np.sqrt(batch_rvar + eps)
            for c_in in range(norms.shape[1]):
                if it in downsample_idxs:
                    count_col_skip = col_counts[it-4]
                    if batchnorms[it] is not None:
                        norm_adjacency_matrix[count_row+chan,count_col_skip+c_in] = norms[chan,c_in] * batch_mult
                    else:
                        norm_adjacency_matrix[count_row+chan,count_col_skip+c_in] = norms[chan,c_in]
                elif it in after_downsample_idxs:
                    count_col_skip = col_counts[it-1]
                    if batchnorms[it] is not None:
                        norm_adjacency_matrix[count_row+chan,count_col+c_in] = norms[chan,c_in] * batch_mult
                        norm_adjacency_matrix[count_row+chan,count_col_skip+c_in] = norms[chan,c_in] * batch_mult
                    else:
                        norm_adjacency_matrix[count_row+chan,count_col+c_in] = norms[chan,c_in]
                        norm_adjacency_matrix[count_row+chan,count_col_skip+c_in] = norms[chan,c_in]
                else:
                    if batchnorms[it] is not None:
                        norm_adjacency_matrix[count_row+chan,count_col+c_in] = norms[chan,c_in] * batch_mult
                    else:
                        norm_adjacency_matrix[count_row+chan,count_col+c_in] = norms[chan,c_in]
        count_row += norms.shape[0]
        count_col += norms.shape[1]
        it += 1

    # Now that we have the matrix, we iteratively extract the strongest
    # multiplicative paths to keep while pruning. This function returns
    # a boolean array with the convolutions that should be pruned.
    pruned = gru.longest_path_prune(norm_adjacency_matrix, perc, skip_connection_matrix)

    ### Perform the pruning on the actual Pytorch model
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

    # Perform redundancy pruning (if active) or prune batch normalization.
    # Redundancy pruning also contains the batch normalization pruning step.
    if Redun:
        Prune_Redundant_Convolutions_ResNet50(pmodel)
    else:
        apply_mask_to_batchnorm_ResNet50(pmodel)
    return pmodel

def IndivL1_Global_ResNet50(pmodel, tot_perc, Redun=True):
    r"""Prune FCN-ResNet50 model using individual filter 
         pruning based on the L1 vector norm.

    Args:
        - pmodel: PyTorch-model of FCN-ResNet50 to be pruned.
        - tot_perc (float): The total fraction of convolutions
            we want pruned at the end of this pruning step.
        - Redun (bool): Whether to perform redundancy pruning
            at the end of the pruning phase. Default is True.
    """
    # Calculate by what percentage the model needs to be pruned
    # to obtain 'tot_perc' percent of pruning. It can e.g. require
    # less than the expected percentage because redundancy pruning
    # removed a significant amount in the previous pruning phase.
    N = 1
    if pruned_before_ResNet50(pmodel):
        prun_conv, tot_conv, frac_prun = fraction_pruned_convs_ResNet50(pmodel)
        if frac_prun >= 1 - tot_perc:
            print("No pruning to be done")
            return pmodel
        perc = 1.0 - tot_perc/(1.0 - frac_prun)
    else:
        perc = 1.0 - tot_perc

    parameters_to_prune = get_convs_ResNet50(pmodel)
    if pruned_before_ResNet50(pmodel) and Redun:
        Prune_Redundant_Convolutions_ResNet50(pmodel)

    # Compute the convolution norms as L1 vector norm.
    all_norms = np.array([], dtype=np.float32)
    layer_norms = []
    for modul, nam in parameters_to_prune:
        orig = getattr(modul, nam)
        norms = compute_Ln_norms_conv(orig, N).numpy()
        all_norms = np.concatenate((all_norms, norms[norms > 0].flatten()))
        layer_norms.append(norms.tolist())
    threshold = np.percentile(all_norms, 100*perc)

    ### Perform the pruning on the actual Pytorch model
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

    # Perform redundancy pruning (if active) or prune batch normalization.
    # Redundancy pruning also contains the batch normalization pruning step.
    if Redun:
        Prune_Redundant_Convolutions_ResNet50(pmodel)
    else:
        apply_mask_to_batchnorm_ResNet50(pmodel)
    return pmodel

def IndivSV_Global_ResNet50(pmodel, tot_perc, Redun=True):
    r"""Prune FCN-ResNet50 model using individual filter 
         pruning based on the spectral operator norm.

    Args:
        - pmodel: PyTorch-model of FCN-ResNet50 to be pruned.
        - tot_perc (float): The total fraction of convolutions
            we want pruned at the end of this pruning step.
        - Redun (bool): Whether to perform redundancy pruning
            at the end of the pruning phase. Default is True.
    """
    # Calculate by what percentage the model needs to be pruned
    # to obtain 'tot_perc' percent of pruning. It can e.g. require
    # less than the expected percentage because redundancy pruning
    # removed a significant amount in the previous pruning phase.
    if pruned_before_ResNet50(pmodel):
        prun_conv, tot_conv, frac_prun = fraction_pruned_convs_ResNet50(pmodel)
        if frac_prun >= 1 - tot_perc:
            print("No pruning to be done")
            return pmodel
        perc = 1.0 - tot_perc/(1.0 - frac_prun)
    else:
        perc = 1.0 - tot_perc

    M = 7 # Size of temporary input image
    order = 'max' # SVD-norm used for pruning
    parameters_to_prune = get_convs_ResNet50(pmodel)
    batchnorms = get_batchnorms_ResNet50(pmodel)

    if pruned_before_ResNet50(pmodel) and Redun:
        Prune_Redundant_Convolutions_ResNet50(pmodel)

    # Compute the convolution norms using Fourier SVD decomposition
    it = 0
    all_norms = np.array([], dtype=np.float32)
    layer_norms = []
    for modul, nam in parameters_to_prune:
        orig = getattr(modul, nam)
        norms = compute_FourierSVD_norms(orig, M, order).numpy()
        if batchnorms[it] is not None:
            bn_mults = batchnorms[it].weight.data
            bn_rvar = batchnorms[it].running_var.data
            eps = batchnorms[it].eps
            for chan in range(norms.shape[0]):
                batch_mult = np.abs(bn_mults[chan].cpu().numpy())
                batch_rvar = bn_rvar[chan].cpu().numpy()
                batch_mult = batch_mult / np.sqrt(batch_rvar + eps)
                norms[chan,:] *= batch_mult
        all_norms = np.concatenate((all_norms, norms[norms > 0].flatten()))
        layer_norms.append(norms.tolist())
        it += 1
    threshold = np.percentile(all_norms, 100*perc)

    ### Perform the pruning on the actual Pytorch model
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

    # Perform redundancy pruning (if active) or prune batch normalization.
    # Redundancy pruning also contains the batch normalization pruning step.
    if Redun:
        Prune_Redundant_Convolutions_ResNet50(pmodel)
    else:
        apply_mask_to_batchnorm_ResNet50(pmodel)
    return pmodel

def LEAN_SV_MSD(pmodel, tot_perc, Redun=True, verbose=True):
    r"""Prune MS-D model using LEAN pruning. 

    Args:
        - pmodel: PyTorch-model of MS-D network to be pruned.
        - tot_perc (float): The total fraction of convolutions
            we want pruned at the end of this pruning step.
        - Redun (bool): Whether to perform redundancy pruning
            at the end of the pruning phase. Default is True.
    """
    # Calculate by what percentage the model needs to be pruned
    # to obtain 'tot_perc' percent of pruning. It can e.g. require
    # less than the expected percentage because redundancy pruning
    # removed a significant amount in the previous pruning phase.
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

    ### Create the norm graph
    arr_size = pmodel.depth + pmodel.c_in + pmodel.c_out
    norm_adjacency_matrix = np.zeros(shape=(arr_size, arr_size), dtype=np.float32)
    count = pmodel.c_in
    for modul, nam in parameters_to_prune:
        orig = getattr(modul, nam)
        if orig.size()[-1] == 1:
            orig1x1 = orig.unsqueeze(2)
            norms = compute_FourierSVD_norms(orig1x1, M, order).numpy()
            for i in range(norms.shape[0]):
                for k in range(norms.shape[1]):
                    norm_adjacency_matrix[count+i,k] = norms[i,k]
        else:
            norms = compute_FourierSVD_norms(orig, M, order).numpy()[0]
            for k in range(norms.shape[0]):
                norm_adjacency_matrix[count,k] = norms[k]
        count += 1

    # Now that we have the matrix, we iteratively extract the strongest
    # multiplicative paths to keep while pruning. This function returns
    # a boolean array with the convolutions that should be pruned.
    pruned = gru.longest_path_prune(norm_adjacency_matrix, perc)

    ### Perform the pruning on the actual Pytorch model
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

    # Perform redundancy pruning (if active) or prune the biases if the entire
    # convolution layer is pruned. Redundancy pruning also contains the bias pruning.
    if Redun:
        Prune_Redundant_Convolutions_MSD(pmodel)
    else:
        prune_biases_MSD(pmodel)
    return pmodel

def IndivSV_Global_MSD(pmodel, tot_perc, Redun=True, verbose=True):
    r"""Prune MS-D model using individual filter pruning
        using the spectral operator norm.

    Args:
        - pmodel: PyTorch-model of MS-D network to be pruned.
        - tot_perc (float): The total fraction of convolutions
            we want pruned at the end of this pruning step.
        - Redun (bool): Whether to perform redundancy pruning
            at the end of the pruning phase. Default is True.
    """
    # Calculate by what percentage the model needs to be pruned
    # to obtain 'tot_perc' percent of pruning. It can e.g. require
    # less than the expected percentage because redundancy pruning
    # removed a significant amount in the previous pruning phase.
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

    # Compute the convolution norms using Fourier SVD decomposition
    all_norms = np.array([], dtype=np.float32)
    layer_norms = []
    for modul, nam in parameters_to_prune:
        orig = getattr(modul, nam)
        if orig.size()[-1] == 1:
            orig1x1 = orig.unsqueeze(2)
            norms = compute_FourierSVD_norms(orig1x1, M, order).numpy()
        else:
            norms = compute_FourierSVD_norms(orig, M, order).numpy()
        all_norms = np.concatenate((all_norms, norms[norms > 0].flatten()))
        layer_norms.append(norms.tolist())
    threshold = np.percentile(all_norms, 100*perc)

    ### Perform the pruning on the actual Pytorch model
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

    # Perform redundancy pruning (if active) or prune the biases if the entire
    # convolution layer is pruned. Redundancy pruning also contains the bias pruning.
    if Redun:
        Prune_Redundant_Convolutions_MSD(pmodel)
    else:
        prune_biases_MSD(pmodel)
    return pmodel

def IndivL1_Global_MSD(pmodel, tot_perc, Redun=True, verbose=True):
    r"""Prune MS-D model using individual filter pruning
        using the L1 vector norm.

    Args:
        - pmodel: PyTorch-model of MS-D network to be pruned.
        - tot_perc (float): The total fraction of convolutions
            we want pruned at the end of this pruning step.
        - Redun (bool): Whether to perform redundancy pruning
            at the end of the pruning phase. Default is True.
    """
    # Calculate by what percentage the model needs to be pruned
    # to obtain 'tot_perc' percent of pruning. It can e.g. require
    # less than the expected percentage because redundancy pruning
    # removed a significant amount in the previous pruning phase.
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

    # Compute the convolution norms using L1 vector norm
    all_norms = np.array([], dtype=np.float32)
    layer_norms = []
    for modul, nam in parameters_to_prune:
        orig = getattr(modul, nam)
        if orig.size()[-1] == 1:
            orig1x1 = orig.unsqueeze(2)
            norms = compute_Ln_norms_conv(orig1x1, N).numpy()
        else:
            norms = compute_Ln_norms_conv(orig, N).numpy()
        all_norms = np.concatenate((all_norms, norms[norms > 0].flatten()))
        layer_norms.append(norms.tolist())
    threshold = np.percentile(all_norms, 100*perc)

    ### Perform the pruning on the actual Pytorch model
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

    # Perform redundancy pruning (if active) or prune the biases if the entire
    # convolution layer is pruned. Redundancy pruning also contains the bias pruning.
    if Redun:
        Prune_Redundant_Convolutions_MSD(pmodel)
    else:
        prune_biases_MSD(pmodel)
    return pmodel

def LEAN_SV_MSD_3x3(pmodel, tot_perc, Redun=True, verbose=True):
    r"""Prune MS-D model using LEAN pruning, excluding the
         final layer of 1x1-convolutions. 

    Args:
        - pmodel: PyTorch-model of MS-D network to be pruned.
        - tot_perc (float): The total fraction of convolutions
            we want pruned at the end of this pruning step.
        - Redun (bool): Whether to perform redundancy pruning
            at the end of the pruning phase. Default is True.
    """
    # Calculate by what percentage the model needs to be pruned
    # to obtain 'tot_perc' percent of pruning. It can e.g. require
    # less than the expected percentage because redundancy pruning
    # removed a significant amount in the previous pruning phase.
    if pruned_before_MSD(pmodel):
        prun_conv, tot_conv, frac_prun = fraction_pruned_convs_MSD3x3(pmodel)
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
    parameters_to_prune = parameters_to_prune[:-1]

    if pruned_before_MSD(pmodel) and Redun:
        Prune_Redundant_Convolutions_MSD3x3(pmodel)

    ### Create the norm graph
    arr_size = pmodel.depth + pmodel.c_in + pmodel.c_out
    norm_adjacency_matrix = np.zeros(shape=(arr_size, arr_size), dtype=np.float32)
    count = pmodel.c_in
    for modul, nam in parameters_to_prune:
        orig = getattr(modul, nam)
        if orig.size()[-1] == 1:
            raise Exception("Error found 1x1 convolution!")
        else:
            norms = compute_FourierSVD_norms(orig, M, order).numpy()[0]
            for k in range(norms.shape[0]):
                norm_adjacency_matrix[count,k] = norms[k]
        count += 1

    # Now that we have the matrix, we iteratively extract the strongest
    # multiplicative paths to keep while pruning. This function returns
    # a boolean array with the convolutions that should be pruned.
    pruned = gru.longest_path_prune(norm_adjacency_matrix, perc)

    ### Perform the pruning on the actual Pytorch model
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

    # Perform redundancy pruning (if active) or prune the biases if the entire
    # convolution layer is pruned. Redundancy pruning also contains the bias pruning.
    if Redun:
        Prune_Redundant_Convolutions_MSD3x3(pmodel)
    else:
        prune_biases_MSD3x3(pmodel)
    return pmodel

def IndivL1_Global_MSD_3x3(pmodel, tot_perc, Redun=True, verbose=True):
    r"""Prune MS-D model using individual filter pruning
        using the L1 vector norm, excluding the final layer
        of 1x1 convolutions.

    Args:
        - pmodel: PyTorch-model of MS-D network to be pruned.
        - tot_perc (float): The total fraction of convolutions
            we want pruned at the end of this pruning step.
        - Redun (bool): Whether to perform redundancy pruning
            at the end of the pruning phase. Default is True.
    """
    # Calculate by what percentage the model needs to be pruned
    # to obtain 'tot_perc' percent of pruning. It can e.g. require
    # less than the expected percentage because redundancy pruning
    # removed a significant amount in the previous pruning phase.
    N = 1 # Order of Ln norm
    if pruned_before_MSD(pmodel):
        prun_conv, tot_conv, frac_prun = fraction_pruned_convs_MSD3x3(pmodel)
        if frac_prun >= 1 - tot_perc:
            if verbose:
                print("No pruning to be done")
            return pmodel
        perc = 1.0 - tot_perc/(1.0 - frac_prun)
    else:
        perc = 1.0 - tot_perc
    parameters_to_prune = get_convs_MSD(pmodel)
    parameters_to_prune = parameters_to_prune[:-1]

    if pruned_before_MSD(pmodel) and Redun:
        Prune_Redundant_Convolutions_MSD3x3(pmodel)

    # Compute the convolution norms using L1 vector norm
    all_norms = np.array([], dtype=np.float32)
    layer_norms = []
    for modul, nam in parameters_to_prune:
        orig = getattr(modul, nam)
        if orig.size()[-1] == 1:
            raise Exception("Error found 1x1 convolution!")
        else:
            norms = compute_Ln_norms_conv(orig, N).numpy()
        all_norms = np.concatenate((all_norms, norms[norms > 0].flatten()))
        layer_norms.append(norms.tolist())
    threshold = np.percentile(all_norms, 100*perc)

    ### Perform the pruning on the actual Pytorch model
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

    # Perform redundancy pruning (if active) or prune the biases if the entire
    # convolution layer is pruned. Redundancy pruning also contains the bias pruning.
    if Redun:
        Prune_Redundant_Convolutions_MSD3x3(pmodel)
    else:
        prune_biases_MSD3x3(pmodel)
    return pmodel

def IndivSV_Global_MSD_3x3(pmodel, tot_perc, Redun=True, verbose=True):
    r"""Prune MS-D model using individual filter pruning
        using the spectral operator norm, excluding the 
        final layer of 1x1 convolutions.

    Args:
        - pmodel: PyTorch-model of MS-D network to be pruned.
        - tot_perc (float): The total fraction of convolutions
            we want pruned at the end of this pruning step.
        - Redun (bool): Whether to perform redundancy pruning
            at the end of the pruning phase. Default is True.
    """
    # Calculate by what percentage the model needs to be pruned
    # to obtain 'tot_perc' percent of pruning. It can e.g. require
    # less than the expected percentage because redundancy pruning
    # removed a significant amount in the previous pruning phase.
    if pruned_before_MSD(pmodel):
        prun_conv, tot_conv, frac_prun = fraction_pruned_convs_MSD3x3(pmodel)
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
    parameters_to_prune = parameters_to_prune[:-1]

    if pruned_before_MSD(pmodel) and Redun:
        Prune_Redundant_Convolutions_MSD3x3(pmodel)

    # Compute the convolution norms using Fourier SVD decomposition
    all_norms = np.array([], dtype=np.float32)
    layer_norms = []
    for modul, nam in parameters_to_prune:
        orig = getattr(modul, nam)
        if orig.size()[-1] == 1:
            raise Exception("Error found 1x1 convolution!")
        else:
            norms = compute_FourierSVD_norms(orig, M, order).numpy()
        all_norms = np.concatenate((all_norms, norms[norms > 0].flatten()))
        layer_norms.append(norms.tolist())
    threshold = np.percentile(all_norms, 100*perc)
    
    ### Perform the pruning on the actual Pytorch model
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

    # Perform redundancy pruning (if active) or prune the biases if the entire
    # convolution layer is pruned. Redundancy pruning also contains the bias pruning.
    if Redun:
        Prune_Redundant_Convolutions_MSD3x3(pmodel)
    else:
        prune_biases_MSD3x3(pmodel)
    return pmodel


##########################################################################
###  Pruning auxiliary functions for batch-norm, redundancy etc. below ###
##########################################################################

def prune_biases_MSD(model):
    r"""
    Prune biases in MS-D model layers if the entire layer is pruned.
    The final_layer biases are for output channels, will never be pruned
    """
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
    r"""Prune redundant convolutions of MS-D model. A convolution
    is labeled as redundant if
        1) all the input convolutions related to it are pruned.
    """
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
    print("Pruned {} redundant convolutions.".format(count))
    prune_biases_MSD(model)

def prune_biases_MSD3x3(model):
    r"""
    Prune biases in MS-D model layers if the entire layer is pruned,
     excluding the final layer of 1x1 convolutions.
    """
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
    r"""Prune redundant convolutions of MS-D model, excluding 
    the final layer. A convolution is labeled as redundant if
        1) all the input convolutions related to it are pruned.
    """
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

def apply_mask_to_batchnorm_ResNet50(pmodel):
    r"""Given the pruned masks of the convolutional layers,
    prune the batch normalization channels if the entire 
    associated convolutional channel has been pruned.
    """
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

def Prune_Redundant_Convolutions_ResNet50(pmodel, bn_thrs = 1e-10):
    r"""Prune redundant convolutions of FCN-ResNet50 model. 
    A convolution is labeled as redundant if
        1) all the input convolutions related to it are pruned.
        2) the running variance of the associated batch normalization
            channel is less than 10^-10.
    """
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

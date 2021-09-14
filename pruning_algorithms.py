import torch
import msd_pytorch as mp
import numpy as np
import torch.nn as nn
import torch.nn.utils.prune as prune

from norm_calculation_methods import *
from pruning_utils import *
import graph_algorithms as gru


##################################
###  RESNET PRUNING FUNCTIONS  ###
##################################

def LEAN_ResNet50(pmodel, tot_perc, Redun=True, verbose=False):
    r"""Prune FCN-ResNet50 model using LEAN pruning.
    NOTE: This function assumes that ResNet has an average pooling layer
            instead of max-pooling.

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
    prunedQ = pruned_before_ResNet50(pmodel)
    if prunedQ:
        prun_conv, tot_conv, frac_prun = fraction_pruned_convs_ResNet50(pmodel)
        if frac_prun >= 1 - tot_perc:
            if verbose:
                print("No pruning to be done")
            return pmodel
        perc = 1.0 - tot_perc/(1.0 - frac_prun)
    else:
        perc = 1.0 - tot_perc
    if verbose:
        print("Frac pruned convs should be:", 1 - tot_perc)

    M = 16 # Size of temporary input image
    order = 'max' # SVD-norm used for pruning
    parameters_to_prune = get_convs_ResNet50(pmodel)
    batchnorms = get_batchnorms_ResNet50(pmodel)

    # These are the indices of the downsample layer and where they point to
    downsample_idxs = [5,15,28,47]
    after_downsample_idxs = [6,16,29,48]
    # These are the other skip connections which are just identity mappings,
    #  when the identity points to a downsample layer, there are several connections
    skip_idxs = np.array([(9,4),(9,5),(12,8),(19,14),(19,15),(22,18),(25,21),(32,27),(32,28),(35,31),(38,34),(41,37),(44,40),(51,46),(51,47)])
    avg_idxs = [1]

    # If we are considering redundancy pruning, we must remove zero-images to avoid
    #  batch-scaling to skew the norms
    if prunedQ and Redun:
        Prune_Redundant_Convolutions_ResNet50(pmodel)

    # Calculate the number of nodes in the graph:
    nr_nodes = pmodel.c_in # start at c_in because of number of input channels
    for modul, nam in parameters_to_prune:
        orig = getattr(modul, nam)
        nr_nodes += orig.size()[0]
    if verbose:
        print("Number of nodes in graph:", nr_nodes)

    # NOTE: Below follows a big chunk of code. It builds the pruning graph
    # Associated with ResNet50-average-pooling. 
    ignore_edges_list = []
    adj_list =[]
    count_row = pmodel.c_in
    count_col = 0
    it = 0
    row_counts = []
    col_counts = []

    strongconvs = 0
    for modul, nam in parameters_to_prune:
        row_counts.append(count_row)
        col_counts.append(count_col)
        orig = getattr(modul, nam)
        strides = modul.stride
        if prunedQ:
            current_mask = get_default_mask(modul, nam)

        ### Compute the operator norms of the convolutions
        norms = compute_FourierSVD_norms(orig, M, order, strides).numpy()

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
            
            # Deal with the downsample skip connections
            if it in downsample_idxs:
                outidx = count_row + chan
                count_col_skip = col_counts[it-4]
                for c_in in range(orig.size()[1]):
                    if prunedQ and current_mask[chan, c_in].sum() == 0:
                        continue
                    val = norms[chan, c_in]
                    if batchnorms[it] is not None:
                        val *= batch_mult
                    if val > 1:
                        strongconvs += 1
                    inidx = count_col_skip + c_in
                    idx_list = len(adj_list)#This is so that we know the order after we perform topological sort
                    edge = [inidx, outidx, val, idx_list]
                    adj_list.append(edge)
                    ignore_edges_list.append([inidx, outidx, False])
            elif it in after_downsample_idxs:
                # Deal with convolutions that have multiple inputs as a result
                #  of a preceding downsampling connection
                outidx = count_row + chan
                count_col_skip = col_counts[it-1]
                for c_in in range(orig.size()[1]):
                    if prunedQ and current_mask[chan, c_in].sum() == 0:
                        continue
                    val = norms[chan, c_in]
                    if batchnorms[it] is not None:
                        val *= batch_mult
                    if val > 1:
                        strongconvs += 1

                    # We have two edges here
                    inidx = count_col + c_in
                    idx_list = len(adj_list)#This is so that we know the order after we perform topological sort
                    edge = [inidx, outidx, val, idx_list]
                    adj_list.append(edge)
                    ignore_edges_list.append([inidx, outidx, False])
            else:
                # Normal convolutional layers
                outidx = count_row + chan
                for c_in in range(orig.size()[1]):
                    if prunedQ and current_mask[chan, c_in].sum() == 0:
                        continue
                    val = norms[chan, c_in]
                    if batchnorms[it] is not None:
                        val *= batch_mult
                    if val > 1:
                        strongconvs += 1
                    inidx = count_col + c_in
                    idx_list = len(adj_list)#This is so that we know the order after we perform topological sort
                    edge = [inidx, outidx, val, idx_list]
                    adj_list.append(edge)
                    if it in avg_idxs:#This layer is average pooling layer
                        ignore_edges_list.append([inidx, outidx, True])
                    else:
                        ignore_edges_list.append([inidx, outidx, False])

        # Incorporate skip connections that are simple identity mappings,
        #  i.e., are implemented as an addition in the forward pass and
        #  have no learnable parameters
        if it in skip_idxs[:,0]: # This layer has an incoming skip connctn
            idx = np.where(skip_idxs[:,0] == it)[0][0]
            modul_in, nam_in = parameters_to_prune[skip_idxs[idx,1]]
            modul_between, nam_between=parameters_to_prune[skip_idxs[idx,1]+1]
            orig_in = getattr(modul_in, nam_in)
            orig_between = getattr(modul_between, nam_between)
            skip_size_row, skip_size_col = orig_in.size()[0], orig_in.size()[1]
            skip_col = count_col - skip_size_col - orig_between.size()[1]
            for chan in range(skip_size_row):
                for cin in range(skip_size_col):
                    outidx = count_row + chan
                    inidx = skip_col + cin
                    val = 1
                    idx_list = len(adj_list)#This is so that we know the order after we perform topological sort
                    edge = [inidx, outidx, val, idx_list]
                    adj_list.append(edge)
                    ignore_edges_list.append([inidx, outidx, True])
        if it in after_downsample_idxs:# Due to the summation that happen in ResNet
            # after certain downsampling steps, we need this unprunable identity edge.
            count_col_skip = col_counts[it-1]
            modul_in, nam_in = parameters_to_prune[it-1]
            orig_in = getattr(modul_in, nam_in)
            for c_in in range(orig_in.size()[1]):
                outidx = count_col + c_in
                val = 1
                inidx = count_col_skip + c_in
                idx_list = len(adj_list)
                edge = [inidx, outidx, val, idx_list]
                adj_list.append(edge)
                ignore_edges_list.append([inidx, outidx, True])
        count_row += norms.shape[0]
        count_col += norms.shape[1]
        it += 1
    print("Convolutions with norms > 1:", strongconvs)

    # Here, we convert the previous matrix to a data structure that the
    #  Rust implementation can use
    adj_list.sort(key=lambda tup: tup[0])
    adjarr = np.array(adj_list)
    codebook = np.array([list(range(adjarr.shape[0])), adjarr[:,3].tolist()], dtype=np.int32).transpose()
    codebook = codebook[np.argsort(codebook[:, 1])]
    codebook[:,[0, 1]] = codebook[:,[1, 0]]
    ignore_edges_list.sort(key=lambda tup: tup[0])
    ignorearr = np.array(ignore_edges_list)

    # Run the fast Rust implementation
    pruned = gru.longest_path_prune_fast(adjarr, perc, ignore_edges_arr=ignorearr)

    # Based on the paths returned, below performs the actual pruning.
    it = 0
    code_iter = 0
    for modul, nam in parameters_to_prune:
        count_row = row_counts[it]
        count_col = col_counts[it]
        orig = getattr(modul, nam)
        mask = torch.ones_like(orig)
        default_mask = get_default_mask(modul, nam)
        if it in avg_idxs:#We do not prune the pooling layer (its 3x3 here!)
            code_iter += int(default_mask.sum()/9)
        else:
            if it in downsample_idxs:
                for i in range(mask.size()[0]):
                    for j in range(mask.size()[1]):
                        if prunedQ and default_mask[i, j].sum() == 0:
                            # it already was pruned, ergo it has no edge
                            continue
                        code = codebook[code_iter]
                        edge_idx = code[1]
                        if not pruned[edge_idx]:
                            mask[i][j] = torch.zeros_like(orig[i][j])
                        code_iter += 1
            elif it in after_downsample_idxs:
                for i in range(mask.size()[0]):
                    for j in range(mask.size()[1]):
                        if prunedQ and default_mask[i, j].sum() == 0:
                            # it already was pruned, ergo it has no edge
                            continue
                        code = codebook[code_iter]
                        edge_idx = code[1]
                        if not pruned[edge_idx]:
                            mask[i][j] = torch.zeros_like(orig[i][j])
                        code_iter += 1
            else:
                for i in range(mask.size()[0]):
                    for j in range(mask.size()[1]):
                        if prunedQ and default_mask[i, j].sum() == 0:
                            # it already was pruned, ergo it has no edge
                            continue
                        code = codebook[code_iter]
                        edge_idx = code[1]
                        if not pruned[edge_idx]:
                            mask[i][j] = torch.zeros_like(orig[i][j])
                        code_iter += 1
            if it in skip_idxs[:,0]: # This layer has an incoming skip connctn
                idx = np.where(skip_idxs[:,0] == it)[0][0]
                modul_in, nam_in = parameters_to_prune[skip_idxs[idx,1]]
                modul_between, nam_between=parameters_to_prune[skip_idxs[idx,1]+1]
                orig_in = getattr(modul_in, nam_in)
                orig_between = getattr(modul_between, nam_between)
                skip_size_row, skip_size_col = orig_in.size()[0], orig_in.size()[1]
                skip_col = count_col - skip_size_col - orig_between.size()[1]
                code_iter += skip_size_row * skip_size_col
            if it in after_downsample_idxs:# Due to the summation that happen in ResNet
                # after certain downsampling steps, we need this unprunable identity edge
                modul_in, nam_in = parameters_to_prune[it-1]
                orig_in = getattr(modul_in, nam_in)
                code_iter += orig_in.size()[1]
        mask *= default_mask.to(dtype=mask.dtype)
        method = prune.CustomFromMask(mask)
        method.apply(modul, nam, mask)
        it += 1

    if Redun:
        Prune_Redundant_Convolutions_ResNet50(pmodel)
    else:
        apply_mask_to_batchnorm_ResNet50(pmodel)
    return pmodel


def IndivL1_Global_ResNet50(pmodel, tot_perc, Redun=True, verbose=False):
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
            if verbose:
                print("No pruning to be done")
            return pmodel
        perc = 1.0 - tot_perc/(1.0 - frac_prun)
    else:
        perc = 1.0 - tot_perc

    parameters_to_prune = get_convs_ResNet50(pmodel)
    if pruned_before_ResNet50(pmodel) and Redun:
        Prune_Redundant_Convolutions_ResNet50(pmodel)

    avg_idxs = [1]# Index of the average pooling laeyr in the list "parameters to prune"
    all_norms = np.array([], dtype=np.float32)
    layer_norms = []
    it = 0
    for modul, nam in parameters_to_prune:
        orig = getattr(modul, nam)
        if it in avg_idxs:
            it += 1
            continue
        # Compute the convolution norms as L1 vector norm.
        norms = compute_Ln_norms_conv(orig, N).numpy()
        if pruned_before_ResNet50(pmodel):
            all_norms = np.concatenate((all_norms, norms[norms > 0].flatten()))
        else:
            all_norms = np.concatenate((all_norms, norms.flatten()))
        layer_norms.append(norms.tolist())
        it += 1
    threshold = np.percentile(all_norms, 100*perc)

    ### Perform the pruning on the actual Pytorch model
    conv_masks = []
    it = 0
    it2 = 0
    for modul, nam in parameters_to_prune:
        orig = getattr(modul, nam)
        lnorms = layer_norms[it2]
        mask = torch.ones_like(orig)
        default_mask = get_default_mask(modul, nam)
        if it not in avg_idxs:
            for i in range(len(lnorms)):
                for j in range(len(lnorms[i])):
                    if lnorms[i][j] < threshold:
                        mask[i][j] = torch.zeros_like(orig[i][j])
            it2 += 1
        it += 1
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

def IndivSV_Global_ResNet50(pmodel, tot_perc, Redun=True, verbose=False):
    r"""Prune FCN-ResNet50 model using individual filter 
         pruning based on the spectral operator norm.

    Args:
        - pmodel: PyTorch-model of FCN-ResNet50 to be pruned.
        - tot_perc (float): The total fraction of convolutions
            we want pruned at the end of this pruning step.
        - Redun (bool): Whether to perform redundancy pruning
            at the end of the pruning phase. Deft4.
    """
    # Calculate by what percentage the model needs to be pruned
    # to obtain 'tot_perc' percent of pruning. It can e.g. require
    # less than the expected percentage because redundancy pruning
    # removed a significant amount in the previous pruning phase.
    if pruned_before_ResNet50(pmodel):
        prun_conv, tot_conv, frac_prun = fraction_pruned_convs_ResNet50(pmodel)
        if frac_prun >= 1 - tot_perc:
            if verbose:
                print("No pruning to be done")
            return pmodel
        perc = 1.0 - tot_perc/(1.0 - frac_prun)
    else:
        perc = 1.0 - tot_perc

    M = 16 # Size of temporary input image
    order = 'max' # SVD-norm used for pruning
    parameters_to_prune = get_convs_ResNet50(pmodel)
    batchnorms = get_batchnorms_ResNet50(pmodel)

    if pruned_before_ResNet50(pmodel) and Redun:
        Prune_Redundant_Convolutions_ResNet50(pmodel)

    # Compute the operator norms using Fourier transforms
    it = 0
    all_norms = np.array([], dtype=np.float32)
    layer_norms = []
    for modul, nam in parameters_to_prune:
        orig = getattr(modul, nam)
        strides = modul.stride
        if it in avg_idxs:
            it += 1
            continue
        norms = compute_FourierSVD_norms(orig, M, order, strides).numpy()
        if batchnorms[it] is not None:
            bn_mults = batchnorms[it].weight.data
            bn_rvar = batchnorms[it].running_var.data
            eps = batchnorms[it].eps
            for chan in range(norms.shape[0]):
                batch_mult = np.abs(bn_mults[chan].cpu().numpy())
                batch_rvar = bn_rvar[chan].cpu().numpy()
                batch_mult = batch_mult / np.sqrt(batch_rvar + eps)
                norms[chan,:] *= batch_mult
        if pruned_before_ResNet50(pmodel):
            all_norms = np.concatenate((all_norms, norms[norms > 0].flatten()))
        else:
            all_norms = np.concatenate((all_norms, norms.flatten()))
        layer_norms.append(norms.tolist())
        it += 1
    threshold = np.percentile(all_norms, 100*perc)

    ### Perform the pruning on the actual Pytorch model
    conv_masks = []
    it = 0
    it2 = 0
    for modul, nam in parameters_to_prune:
        orig = getattr(modul, nam)
        lnorms = layer_norms[it2]
        mask = torch.ones_like(orig)
        default_mask = get_default_mask(modul, nam)
        if it not in avg_idxs:
            for i in range(len(lnorms)):
                for j in range(len(lnorms[i])):
                    if lnorms[i][j] < threshold:
                        mask[i][j] = torch.zeros_like(orig[i][j])
            it2 += 1
        it += 1
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

#################################
###  U-NET PRUNING FUNCTIONS  ###
#################################

def LEAN_UNet4(pmodel, tot_perc, Redun=True, verbose=False):
    r"""Prune FCN-UNet4 model using LEAN pruning.
    NOTE: This function assumes that U-Net has an average pooling layer
            instead of max-pooling.

    Args:
        - pmodel: PyTorch-model of FCN-UNet4 to be pruned.
        - tot_perc (float): The total fraction of convolutions
            we want pruned at the end of this pruning step.
        - Redun (bool): Whether to perform redundancy pruning
            at the end of the pruning phase. Default is True.
    """
    # Calculate by what percentage the model needs to be pruned
    # to obtain 'tot_perc' percent of pruning. It can e.g. require
    # less than the expected percentage because redundancy pruning
    # removed a significant amount in the previous pruning phase.
    prunedQ = pruned_before_UNet4(pmodel)
    if prunedQ:
        prun_conv, tot_conv, frac_prun = fraction_pruned_convs_UNet4(pmodel)
        if frac_prun >= 1 - tot_perc:
            if verbose:
                print("No pruning to be done")
            return pmodel
        perc = 1.0 - tot_perc/(1.0 - frac_prun)
    else:
        perc = 1.0 - tot_perc
    if verbose:
        print("Frac pruned convs should be:", 1 - tot_perc)

    M = 16 # Size of temporary input image
    order = 'max' # SVD-norm used for pruning
    parameters_to_prune = get_convs_UNet4(pmodel)
    batchnorms = get_batchnorms_UNet4(pmodel)

    # These are the indices of the downsample layer and where they point to
    # These are the other skip connections which are just identity mappings,
    #  when the identity points to a downsample layer, there are several connections
    after_upscale_idxs = [15,18,21,24]
    skip_idxs = np.array([(15,10),(18,7),(21,4),(24,1)])
    avg_idxs = [2, 5, 8, 11]

    # If we are considering redundancy pruning, we must remove zero-images
    #to avoid batch-scaling to skew the norms
    if prunedQ and Redun:
        Prune_Redundant_Convolutions_UNet4(pmodel)

    # Count the number of nodes that will be in the pruning graph
    nr_nodes = pmodel.c_in # start at c_in because of number of input channels
    for modul, nam in parameters_to_prune:
        orig = getattr(modul, nam)
        nr_nodes += orig.size()[0]
    print("Number of nodes in graph:", nr_nodes)


    # BELOW FOLLOWS a lot of code to define the pruning graph with
    #  with the different skip connections and pooling etc. 
    ignore_edges_list = []
    adj_list = []
    count_row = pmodel.c_in
    count_col = 0
    it = 0
    row_counts = []
    col_counts = []

    strongconvs = 0
    for modul, nam in parameters_to_prune:
        row_counts.append(count_row)
        col_counts.append(count_col)
        orig = getattr(modul, nam)
        strides = modul.stride
        if prunedQ:
            current_mask = get_default_mask(modul, nam)

        # Calculate operator norms for convolution layer
        norms = compute_FourierSVD_norms(orig, M, order, strides).numpy()

        # Incorporate batch normalization
        if batchnorms[it] is not None:
            bn_mults = batchnorms[it].weight.data
            bn_rvar = batchnorms[it].running_var.data
            eps = batchnorms[it].eps
        if it in after_upscale_idxs:#We need special procedure to deal with skip connections
            idx = np.where(skip_idxs[:,0] == it)[0][0]
            inputidx = skip_idxs[idx,1]
            count_col_skip = col_counts[inputidx]
            modul_in, nam_in = parameters_to_prune[skip_idxs[idx,1]]
            orig_in = getattr(modul_in, nam_in)
            skip_size_row, skip_size_col = orig_in.size()[0],orig_in.size()[1]
            for chan in range(orig.size()[0]):
                if batchnorms[it] is not None:
                    batch_mult = np.abs(bn_mults[chan].cpu().numpy())
                    batch_rvar = bn_rvar[chan].cpu().numpy()
                    batch_mult = batch_mult / np.sqrt(batch_rvar + eps)
                outidx = count_row + chan
                for c_in in range(orig.size()[1]):
                    if prunedQ and current_mask[chan, c_in].sum() == 0:
                        continue
                    if c_in < skip_size_col:
                        inidx = count_col_skip + c_in
                    else:# The output from up1.up, i.e. previous, is last in the concatenation
                        inidx = count_col + c_in - skip_size_col
                    val = norms[chan, c_in]
                    if batchnorms[it] is not None:
                        val *= batch_mult
                    if val > 1:
                        strongconvs += 1
                    idx_list = len(adj_list)#This is so that we know the order after we perform topological sort
                    edge = [inidx, outidx, val, idx_list]
                    adj_list.append(edge)
                    ignore_edges_list.append([inidx, outidx, False])
            count_col -= skip_size_col#norm array is too large because it
            # also contains the concatenated skip connection norms.
        else:
            for chan in range(orig.size()[0]):
                if batchnorms[it] is not None:
                    batch_mult = np.abs(bn_mults[chan].cpu().numpy())
                    batch_rvar = bn_rvar[chan].cpu().numpy()
                    batch_mult = batch_mult / np.sqrt(batch_rvar + eps)
                outidx = count_row + chan
                for c_in in range(orig.size()[1]):
                    if prunedQ and current_mask[chan, c_in].sum() == 0:
                        continue
                    val = norms[chan, c_in]
                    if batchnorms[it] is not None:
                        val *= batch_mult
                    if val > 1:
                        strongconvs += 1
                    inidx = count_col + c_in
                    idx_list = len(adj_list)#This is so that we know the order after we perform topological sort
                    edge = [inidx, outidx, val, idx_list]
                    adj_list.append(edge)
                    if it in avg_idxs:#This layer is average pooling layer
                        ignore_edges_list.append([inidx, outidx, True])
                    else:
                        ignore_edges_list.append([inidx, outidx, False])
        count_row += norms.shape[0]
        count_col += norms.shape[1]
        it += 1
    if verbose:
        print("Convolutions with norms > 1:", strongconvs)

    # Here, we convert the previous matrix to a data structure that the
    #  Rust implementation can use
    adj_list.sort(key=lambda tup: tup[0])
    adjarr = np.array(adj_list)
    codebook = np.array([list(range(adjarr.shape[0])), adjarr[:,3].tolist()], dtype=np.int32).transpose()
    codebook = codebook[np.argsort(codebook[:, 1])]
    codebook[:,[0, 1]] = codebook[:,[1, 0]]
    ignore_edges_list.sort(key=lambda tup: tup[0])
    ignorearr = np.array(ignore_edges_list)

    # Run the fast Rust implementation
    pruned = gru.longest_path_prune_fast(adjarr, perc, ignore_edges_arr=ignorearr)

    # Based on the paths returned, below performs the actual pruning.
    it = 0
    code_iter = 0
    for modul, nam in parameters_to_prune:
        count_row = row_counts[it]
        count_col = col_counts[it]
        orig = getattr(modul, nam)
        mask = torch.ones_like(orig)
        default_mask = get_default_mask(modul, nam)
        if it in avg_idxs:
            code_iter += int(default_mask.sum()/4)
        else:
            if it in after_upscale_idxs:#We need special procedure to deal with skip concts
                for i in range(mask.size()[0]):
                    for j in range(mask.size()[1]):
                        if prunedQ and default_mask[i, j].sum() == 0:
                            # it already was pruned, ergo it has no edge
                            continue
                        code = codebook[code_iter]
                        edge_idx = code[1]
                        if not pruned[edge_idx]:
                            mask[i][j] = torch.zeros_like(orig[i][j])
                        code_iter += 1
            else:
                for i in range(mask.size()[0]):
                    for j in range(mask.size()[1]):
                        if prunedQ and default_mask[i, j].sum() == 0:
                            # it already was pruned, ergo it has no edge
                            continue
                        code = codebook[code_iter]
                        edge_idx = code[1]
                        if not pruned[edge_idx]:
                            mask[i][j] = torch.zeros_like(orig[i][j])
                        code_iter += 1
        mask *= default_mask.to(dtype=mask.dtype)
        method = prune.CustomFromMask(mask)
        method.apply(modul, nam, mask)
        it += 1

    if Redun:
        Prune_Redundant_Convolutions_UNet4(pmodel)
    else:
        apply_mask_to_batchnorm_UNet4(pmodel)
    return pmodel

def IndivL1_Global_UNet4(pmodel, tot_perc, Redun=True, verbose=False):
    r"""Prune FCN-UNet4 model using individual filter 
         pruning based on the L1 vector norm.

    Args:
        - pmodel: PyTorch-model of FCN-UNet4 to be pruned.
        - tot_perc (float): The total fraction of convolutions
            we want pruned at the end of this pruning step.
        - Redun (bool): Whether to perform redundancy pruning
            at the end of the pruning phase. Default is True.
    """
    # Vector norm to use:
    N = 1

    # Calculate by what percentage the model needs to be pruned
    # to obtain 'tot_perc' percent of pruning. It can e.g. require
    # less than the expected percentage because redundancy pruning
    # removed a significant amount in the previous pruning phase.
    if pruned_before_UNet4(pmodel):
        prun_conv, tot_conv, frac_prun = fraction_pruned_convs_UNet4(pmodel)
        if frac_prun >= 1 - tot_perc:
            if verbose:
                print("No pruning to be done")
            return pmodel
        perc = 1.0 - tot_perc/(1.0 - frac_prun)
    else:
        perc = 1.0 - tot_perc
    if verbose:
        print("Frac pruned convs should be:", 1 - tot_perc)

    parameters_to_prune = get_convs_UNet4(pmodel)
    if pruned_before_UNet4(pmodel) and Redun:
        Prune_Redundant_Convolutions_UNet4(pmodel)

    avg_idxs = [2, 5, 8, 11]
    all_norms = np.array([], dtype=np.float32)
    layer_norms = []
    it = 0
    for modul, nam in parameters_to_prune:
        orig = getattr(modul, nam)
        if it in avg_idxs:
            it += 1
            continue
        norms = compute_Ln_norms_conv(orig, N).numpy()
        if pruned_before_UNet4(pmodel):
            all_norms = np.concatenate((all_norms, norms[norms > 0].flatten()))
        else:
            all_norms = np.concatenate((all_norms, norms.flatten()))
        layer_norms.append(norms.tolist())
        it += 1
    threshold = np.percentile(all_norms, 100*perc)

    ### Perform the pruning on the actual Pytorch model
    conv_masks = []
    it = 0
    it2 = 0
    for modul, nam in parameters_to_prune:
        orig = getattr(modul, nam)
        lnorms = layer_norms[it2]
        mask = torch.ones_like(orig)
        default_mask = get_default_mask(modul, nam)
        if it in avg_idxs:
            pass
        else:
            for i in range(len(lnorms)):
                for j in range(len(lnorms[i])):
                    if lnorms[i][j] < threshold:
                        mask[i][j] = torch.zeros_like(orig[i][j])
            it2 += 1
        it += 1
        mask *= default_mask.to(dtype=mask.dtype)
        method = prune.CustomFromMask(mask)
        method.apply(modul, nam, mask)

    if Redun:
        Prune_Redundant_Convolutions_UNet4(pmodel)
    else:
        apply_mask_to_batchnorm_UNet4(pmodel)
    return pmodel

def IndivSV_Global_UNet4(pmodel, tot_perc, Redun=True, verbose=False):
    r"""Prune FCN-UNet4 model using individual filter 
         pruning based on the spectral operator norm.

    Args:
        - pmodel: PyTorch-model of FCN-UNet4 to be pruned.
        - tot_perc (float): The total fraction of convolutions
            we want pruned at the end of this pruning step.
        - Redun (bool): Whether to perform redundancy pruning
            at the end of the pruning phase. Default is True.
    """
    # Calculate by what percentage the model needs to be pruned
    # to obtain 'tot_perc' percent of pruning. It can e.g. require
    # less than the expected percentage because redundancy pruning
    # removed a significant amount in the previous pruning phase.
    if pruned_before_UNet4(pmodel):
        prun_conv, tot_conv, frac_prun = fraction_pruned_convs_UNet4(pmodel)
        if frac_prun >= 1 - tot_perc:
            if verbose:
                print("No pruning to be done")
            return pmodel
        perc = 1.0 - tot_perc/(1.0 - frac_prun)
    else:
        perc = 1.0 - tot_perc
    if verbose:
        print("Frac pruned convs should be:", 1 - tot_perc)

    M = 16 # Size of temporary input image
    order = 'max' # SVD-norm used for pruning
    parameters_to_prune = get_convs_UNet4(pmodel)
    batchnorms = get_batchnorms_UNet4(pmodel)
    if pruned_before_UNet4(pmodel) and Redun:
        Prune_Redundant_Convolutions_UNet4(pmodel)
    avg_idxs = [2, 5, 8, 11]

    it = 0
    all_norms = np.array([], dtype=np.float32)
    layer_norms = []
    for modul, nam in parameters_to_prune:
        orig = getattr(modul, nam)
        strides = modul.stride
        if it in avg_idxs:
            it += 1
            continue
        norms = compute_FourierSVD_norms(orig, M, order, strides).numpy()
        if batchnorms[it] is not None:
            bn_mults = batchnorms[it].weight.data
            bn_rvar = batchnorms[it].running_var.data
            eps = batchnorms[it].eps
            for chan in range(norms.shape[0]):
                batch_mult = np.abs(bn_mults[chan].cpu().numpy())
                batch_rvar = bn_rvar[chan].cpu().numpy()
                batch_mult = batch_mult / np.sqrt(batch_rvar + eps)
                norms[chan,:] *= batch_mult
        if pruned_before_UNet4(pmodel):
            all_norms = np.concatenate((all_norms, norms[norms > 0].flatten()))
        else:
            all_norms = np.concatenate((all_norms, norms.flatten()))
        layer_norms.append(norms.tolist())
        it += 1
    threshold = np.percentile(all_norms, 100*perc)

    ### Perform the pruning on the actual Pytorch model
    masks = []
    it = 0
    it2 = 0
    for modul, nam in parameters_to_prune:
        orig = getattr(modul, nam)
        lnorms = layer_norms[it2]
        mask = torch.ones_like(orig)
        default_mask = get_default_mask(modul, nam)
        if it in avg_idxs:
            pass
        else:
            for i in range(len(lnorms)):
                for j in range(len(lnorms[i])):
                    if lnorms[i][j] < threshold:
                        mask[i][j] = torch.zeros_like(orig[i][j])
            it2 += 1
        it += 1
        mask *= default_mask.to(dtype=mask.dtype)
        method = prune.CustomFromMask(mask)
        method.apply(modul, nam, mask)
    if Redun:
        Prune_Redundant_Convolutions_UNet4(pmodel)
    else:
        apply_mask_to_batchnorm_UNet4(pmodel)
    return pmodel


################################
###  MS-D PRUNING FUNCTIONS  ###
################################


def LEAN_MSD(pmodel, tot_perc, Redun=True, verbose=False):
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
    M = 128 # Size of temporary input image, (larger because of dilations)
    order = 'max' # SVD-norm used for pruning
    parameters_to_prune = get_convs_MSD(pmodel)

    if pruned_before_MSD(pmodel) and Redun:
        Prune_Redundant_Convolutions_MSD(pmodel)

    ### Create the norm graph
    count = pmodel.c_in
    adj_list = []
    for modul, nam in parameters_to_prune:
        orig = getattr(modul, nam)
        strides = (1,1)
        if prunedQ:
            current_mask = get_default_mask(modul, nam)
        if orig.size()[-1] == 1:
            # Deal with the final layer of 1x1-convolutions
            orig1x1 = orig.unsqueeze(2)
            # Compute operator norm:
            norms = compute_FourierSVD_norms(orig1x1, M, order, strides).numpy()

            for i in range(orig1x1.size()[0]):
                outidx = count + i
                for inidx in range(orig1x1.size()[1]):
                    if prunedQ and current_mask[i, inidx].sum() == 0:
                        continue
                    else:
                        val = norms[i, inidx]
                        idx_list = len(adj_list)#This is so that we know the order after we perform topological sort
                        edge = [inidx, outidx, val, idx_list]
                        adj_list.append(edge)
        else:
            # Deal with all the other regular 3x3-convolutions
            # Compute operator norm:
            norms = compute_FourierSVD_norms(orig, M, order, strides).numpy()[0]

            for i in range(orig.size()[0]):
                outidx = count + i
                for inidx in range(orig.size()[1]):
                    if prunedQ and current_mask[i, inidx].sum() == 0:
                        continue
                    else:
                        val = norms[inidx]
                        idx_list = len(adj_list)#This is so that we know the order after we perform topological sort
                        edge = [inidx, outidx, val, idx_list]
                        adj_list.append(edge)
        count += 1
    adj_list.sort(key=lambda tup: tup[0])
    adjarr = np.array(adj_list)
    codebook = np.array([list(range(adjarr.shape[0])), adjarr[:,3].tolist()], dtype=np.int32).transpose()
    codebook = codebook[np.argsort(codebook[:, 1])]
    codebook[:,[0, 1]] = codebook[:,[1, 0]]

    # Run the fast Rust implementation
    pruned = gru.longest_path_prune_fast(adjarr[:,:3], perc, ignore_edges_arr=None)

    ### Perform the pruning on the actual Pytorch model
    code_iter = 0
    for modul, nam in parameters_to_prune:
        orig = getattr(modul, nam)
        mask = torch.ones_like(orig)
        default_mask = get_default_mask(modul, nam)
        for i in range(mask.size()[0]):
            for inidx in range(mask.size()[1]):
                if prunedQ and default_mask[i, inidx].sum() == 0:# it already was pruned, ergo it has no edge
                    continue
                code = codebook[code_iter]
                edge_idx = code[1]
                if not pruned[edge_idx]:
                    mask[i][inidx] = torch.zeros_like(orig[i][inidx])
                code_iter += 1
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

def IndivSV_Global_MSD(pmodel, tot_perc, Redun=True, verbose=False):
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
    
    M = 128 # Size of temporary input image, (larger because of dilations)
    order = 'max' # SVD-norm used for pruning
    parameters_to_prune = get_convs_MSD(pmodel)

    if pruned_before_MSD(pmodel) and Redun:
        Prune_Redundant_Convolutions_MSD(pmodel)

    # Compute the convolution norms using Fourier SVD decomposition
    all_norms = np.array([], dtype=np.float32)
    layer_norms = []
    for modul, nam in parameters_to_prune:
        orig = getattr(modul, nam)
        strides = (1,1)
        if orig.size()[-1] == 1:
            orig1x1 = orig.unsqueeze(2)
            norms = compute_FourierSVD_norms(orig1x1, M, order, strides).numpy()
        else:
            norms = compute_FourierSVD_norms(orig, M, order, strides).numpy()
        if pruned_before_MSD(pmodel):
            all_norms = np.concatenate((all_norms, norms[norms > 0].flatten()))
        else:
            all_norms = np.concatenate((all_norms, norms.flatten()))
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

def IndivL1_Global_MSD(pmodel, tot_perc, Redun=True, verbose=False):
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
        if pruned_before_MSD(pmodel):
            all_norms = np.concatenate((all_norms, norms[norms > 0].flatten()))
        else:
            all_norms = np.concatenate((all_norms, norms.flatten()))
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

def LEAN_MSD_3x3(pmodel, tot_perc, Redun=True, verbose=False):
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
    M = 128 # Size of temporary input image, (larger because of dilations)
    order = 'max' # SVD-norm used for pruning
    parameters_to_prune = get_convs_MSD(pmodel)
    parameters_to_prune = parameters_to_prune[:-1]

    if pruned_before_MSD(pmodel) and Redun:
        Prune_Redundant_Convolutions_MSD3x3(pmodel)

    ### Create the norm graph
    count = pmodel.c_in
    adj_list = []
    for modul, nam in parameters_to_prune:
        orig = getattr(modul, nam)
        if orig.size()[-1] == 1:
            raise Exception("Something went wrong")
        else:
            strides = (1,1)
            if pruned_before_MSD(pmodel):
                current_mask = get_default_mask(modul, nam)

            # Compute convolution operator norm
            norms = compute_FourierSVD_norms(orig, M, order, strides).numpy()[0]
            for i in range(orig.size()[0]):
                outidx = count + i
                for inidx in range(orig.size()[1]):
                    if pruned_before_MSD(pmodel) and current_mask[i, inidx].sum() == 0:
                        continue
                    else:
                        val = norms[inidx]
                        idx_list = len(adj_list)#This is so that we know the order after we perform topological sort
                        edge = [inidx, outidx, val, idx_list]
                        adj_list.append(edge)
        count += 1
    adj_list.sort(key=lambda tup: tup[0])
    adjarr = np.array(adj_list)
    codebook = np.array([list(range(adjarr.shape[0])), adjarr[:,3].tolist()], dtype=np.int32).transpose()
    codebook = codebook[np.argsort(codebook[:, 1])]
    codebook[:,[0, 1]] = codebook[:,[1, 0]]

    # Run fast Rust implementation
    pruned = gru.longest_path_prune_fast(adjarr, perc, ignore_edges_arr=None)

    ### Perform the pruning on the actual Pytorch model
    code_iter = 0
    for modul, nam in parameters_to_prune:
        orig = getattr(modul, nam)
        mask = torch.ones_like(orig)
        default_mask = get_default_mask(modul, nam)
        for i in range(mask.size()[0]):
            for inidx in range(mask.size()[1]):
                if pruned_before_MSD(pmodel) and default_mask[i, inidx].sum() == 0:# it already was pruned, ergo it has no edge
                    continue
                code = codebook[code_iter]
                edge_idx = code[1]
                if not pruned[edge_idx]:
                    mask[i][inidx] = torch.zeros_like(orig[i][inidx])
                code_iter += 1
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

def IndivL1_Global_MSD_3x3(pmodel, tot_perc, Redun=True, verbose=False):
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
            raise Exception("Something went wrong")
            orig1x1 = orig.unsqueeze(2)
            norms = compute_Ln_norms_conv(orig1x1, N).numpy()
        else:
            norms = compute_Ln_norms_conv(orig, N).numpy()
        if pruned_before_MSD(pmodel):
            all_norms = np.concatenate((all_norms, norms[norms > 0].flatten()))
        else:
            all_norms = np.concatenate((all_norms, norms.flatten()))
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

def IndivSV_Global_MSD_3x3(pmodel, tot_perc, Redun=True, verbose=False):
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

    M = 128 # Size of temporary input image, (larger because of dilations)
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
            raise Exception("Something went wrong")
        else:
            strides = (1,1)
            norms = compute_FourierSVD_norms(orig, M, order, strides).numpy()
        if pruned_before_MSD(pmodel):
            all_norms = np.concatenate((all_norms, norms[norms > 0].flatten()))
        else:
            all_norms = np.concatenate((all_norms, norms.flatten()))
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


####  MS-D  ####

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

def Prune_Redundant_Convolutions_MSD(model, verbose=False):
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
    if verbose:
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

def Prune_Redundant_Convolutions_MSD3x3(model, verbose=False):
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
    if verbose:
        print("Pruned {} redundant convolutions.".format(count))
    prune_biases_MSD3x3(model)


####  ResNet50  ####

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

def Prune_Redundant_Convolutions_ResNet50(pmodel, bn_thrs = 1e-45, verbose=False):
    r"""Prune redundant convolutions of FCN-ResNet50 model. 
    A convolution is labeled as redundant if
        1) all the input convolutions related to it are pruned.
        2) the running variance of the associated batch normalization
            channel is less than 10^-10.

    NOTE: This function assumes that ResNet has an average pooling layer
            instead of max-pooling.
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
        if it in [5,15,28,47]: # Downsample layer:
            prev_mask = conv_masks[it-4]
            mask = conv_masks[it]
            for i in range(prev_mask.size()[0]):
                if prev_mask[i].sum() == 0:
                    if mask[:,i].sum() != 0:
                        count += 1
                    mask[:,i] = 0
        elif it in [6,16,29,48]: # Layer after downsample layer:
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
    if verbose:
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


####  U-Net4  ####

def apply_mask_to_batchnorm_UNet4(pmodel):
    r"""Given the pruned masks of the convolutional layers,
    prune the batch normalization channels if the entire 
    associated convolutional channel has been pruned.
    """
    conv_masks = get_conv_masks_UNet4(pmodel)
    names = ['weight','bias']
    batchnorms_to_prune = get_batchnorms_UNet4(pmodel)
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

def Prune_Redundant_Convolutions_UNet4(pmodel, bn_thrs=1e-10, verbose=False):
    r"""Prune redundant convolutions of FCN-UNet4 model. 
    A convolution is labeled as redundant if
        1) all the input convolutions related to it are pruned.
        2) the running variance of the associated batch normalization
            channel is less than 10^-10.

    NOTE: This function assumes that U-Net has average pooling layers
            instead of max-pooling.
    """
    parameters_to_prune = get_convs_UNet4(pmodel)
    conv_masks = get_conv_masks_UNet4(pmodel)
    batchnorms = get_batchnorms_UNet4(pmodel)

    # In this loop, we check for each output channel of each layer if all its input channels
    # are pruned. If so, that channel is pruned and afterwards the accompanying batchnorm.
    it = 0
    count = 0
    #NOTE: While the unet.up1.up may perform the upscaling (indices 14, 17, 20, 23),
    #       it only has one input, the previous layer. However, the convolutions
    #       that come directly after, i.e. unet.up1.conv.conv[0], takes two inputs.
    for modul, nam in parameters_to_prune: #This is only to throw error if misaligned lists
        if it == 0:
            it += 1
            continue
        elif it == 15: # Certain upscaling layers have 2 inputs (see NOTE):
            prev_mask1 = conv_masks[10] #it=10 --> unet.down3.mpconv[0].conv[3]
            prev_mask2 = conv_masks[14] #it=14 --> unet.up1.up
            mask = conv_masks[it]
            p1_col_size = prev_mask1.size()[1]
            for i in range(prev_mask1.size()[0]):
                if prev_mask1[i].sum() == 0:
                    if mask[:,i].sum() != 0:
                        count += 1
                    mask[:,i] = 0
            for i in range(prev_mask2.size()[0]):
                if prev_mask2[i].sum() == 0:
                    if mask[:, p1_col_size + i].sum() != 0:
                        count += 1
                    mask[:, p1_col_size + i] = 0
        elif it == 18: # Certain upscaling layers have 2 inputs (see NOTE):
            prev_mask1 = conv_masks[7] #it=7 --> unet.down2.mpconv[0].conv[3]
            prev_mask2 = conv_masks[17] #it=17 --> unet.up2.up
            mask = conv_masks[it]
            p1_col_size = prev_mask1.size()[1]
            for i in range(prev_mask1.size()[0]):
                if prev_mask1[i].sum() == 0:
                    if mask[:,i].sum() != 0:
                        count += 1
                    mask[:,i] = 0
            for i in range(prev_mask2.size()[0]):
                if prev_mask2[i].sum() == 0:
                    if mask[:, p1_col_size + i].sum() != 0:
                        count += 1
                    mask[:, p1_col_size + i] = 0
        elif it == 21: # Certain upscaling layers have 2 inputs (see NOTE):
            prev_mask1 = conv_masks[4] #it=4 --> unet.down1.mpconv[0].conv[3]
            prev_mask2 = conv_masks[20] #it=20 --> unet.up3.up
            mask = conv_masks[it]
            p1_col_size = prev_mask1.size()[1]
            for i in range(prev_mask1.size()[0]):
                if prev_mask1[i].sum() == 0:
                    if mask[:,i].sum() != 0:
                        count += 1
                    mask[:,i] = 0
            for i in range(prev_mask2.size()[0]):
                if prev_mask2[i].sum() == 0:
                    if mask[:, p1_col_size + i].sum() != 0:
                        count += 1
                    mask[:, p1_col_size + i] = 0
        elif it == 24: # Certain upscaling layers have 2 inputs (see NOTE):
            prev_mask1 = conv_masks[1] #it=1 --> unet.inc.conv.conv[3]
            prev_mask2 = conv_masks[23] #it=23 --> unet.up4.up
            mask = conv_masks[it]
            p1_col_size = prev_mask1.size()[1]
            for i in range(prev_mask1.size()[0]):
                if prev_mask1[i].sum() == 0:
                    if mask[:,i].sum() != 0:
                        count += 1
                    mask[:,i] = 0
            for i in range(prev_mask2.size()[0]):
                if prev_mask2[i].sum() == 0:
                    if mask[:, p1_col_size + i].sum() != 0:
                        count += 1
                    mask[:, p1_col_size + i] = 0
        else:
            prev_mask = conv_masks[it-1]
            mask = conv_masks[it]
            for i in range(prev_mask.size()[0]):
                if prev_mask[i].sum() == 0:
                    if mask[:,i].sum() != 0:
                        count += 1
                    mask[:,i] = 0
        it += 1
    apply_mask_to_batchnorm_UNet4(pmodel)
    if verbose:
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
    apply_mask_to_batchnorm_UNet4(pmodel)


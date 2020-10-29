import torch
import msd_pytorch as mp
from torch.utils.data import DataLoader
import numpy as np
from timeit import default_timer as timer
import torch.nn as nn
import torch.nn.utils.prune as prune
import math
import random
import scipy.linalg
import scipy.sparse

def fourier_operatornorm(kern, n, order):
    r"""Computes the FFT of the filter matrix to efficiently 
        determine the SVD.

    From the paper: The Singular Values of Convolutional Layers, 
    by Hanie Sedghi, Vineet Gupta, Philip M. Long

    Args:
        - kern (torch.Tensor): Convolutional filter matrix.
        - n (int): dimension of the (mock) input image.
        - order (string): String describing the operator norm.
            Either 'max' for spectral, 'nuc' for nuclear, or
            'fro' for Frobenius.

    Returns: norm of filter (float).
    """
    filt = kern.detach().cpu().numpy()
    fourier_coeff = np.fft.fft2(filt, [n,n], axes=[0, 1])
    if order == 'max':
        return np.max(np.abs(fourier_coeff))
    else:
        S = np.sort(np.absolute(fourier_coeff.flatten()))[::-1]
        if order == 'nuc':
            return np.sum(S)
        elif order == 'fro':
            frob = np.sum(S**2)
            return np.sqrt(frob)
        else:
            raise Exception("Not implemented order")

def compute_FourierSVD_norms(t, M, orde):
    r"""Computes the norm of a convolutional layer as interpreted
        as a matrix operator.

    Args:
        t (torch.Tensor): tensor representing the parameter to prune
        M (int): Dimension of (mock) input image.
        orde (string, int): Order of the norm. Either a string 'nuc'
            or 'fro', or an integer.

    Returns:
        norm (list(float)): List of list of norms for all channels,
            and all inputs for the tensor.
    """
    if t.size()[-1] >= 1  and t.size()[-2] >= 1:
        tnorms = []
        for x in range(t.size()[0]):
            normsx = []
            for y in range(t[x].size()[0]):
                kernel = t[x][y]
                knorm = fourier_operatornorm(kernel, M, orde)
                normsx.append(knorm)
            tnorms.append(normsx)
        tnorms = torch.Tensor(tnorms)
    else:
        raise Exception("Asked to compute norm of non-convolutional layer!")
    return tnorms

def compute_Ln_norms_conv(t, N):
    r"""Computes the norm of a convolutional filter as interpreted
        as a kxk vector, as opposed to a matrix operator.

    Args:
        t (torch.Tensor): tensor representing the parameter to prune
        N (string, int): Order of the norm. Either a string 'nuc'
            or 'fro', or an integer.

    Returns:
        norm (list(float)): List of list of norms for all channels,
            and all inputs for the tensor.
    """
    if t.size()[-1] >= 1 and t.size()[-2] >= 1:
        tnorms = []
        for x in range(t.size()[0]):
            normsx = []
            for y in range(t[x].size()[0]):
                kernel = t[x][y]
                if N == 'nuc':
                    knorm = torch_nuc_norm(kernel)
                elif N == 'fro':
                    knorm = torch_fro_norm(kernel)
                elif isinstance(N, int):
                    knorm = torch_norm(kernel, N, None)
                else:
                    raise Exception("Order not implemented")
                normsx.append(knorm)
            tnorms.append(normsx)
        tnorms = torch.Tensor(tnorms)
    else:
        raise Exception("Asked to compute norm of non-convolutional layer!")
    return tnorms

def torch_norm(t, n, dim):
    r"""COPY of PyTorch norm implementation!

    Compute the L_n-norm across all entries in tensor `t` along all dimension 
    except for the one identified by dim.
    Example: if `t` is of shape, say, 3x2x4 and dim=2 (the last dim),
    then norm will have Size [4], and each entry will represent the 
    `L_n`-norm computed using the 3x2=6 entries for each of the 4 channels.

    Args:
        t (torch.Tensor): tensor representing the parameter to prune
        n (int, float, inf, -inf, 'fro', 'nuc'): See documentation of valid
            entries for argument p in torch.norm
        dim (int): dim identifying the channels to prune

    Returns:
        norm (torch.Tensor): L_n norm computed across all dimensions except
            for `dim`. By construction, `norm.shape = t.shape[-1]`.
    """
    dims = list(range(t.dim()))
    if dim is not None:
        if dim < 0:
            dim = dims[dim]
        dims.remove(dim)
    norm = torch.norm(t, p=n, dim=dims)
    return norm

def torch_fro_norm(t):
    r"""Compute the Frobenius-norm across all entries in
        tensor `t` along all dimensions.
    """
    dims = list(range(t.dim()))
    norm = torch.norm(t.cpu(), p='fro', dim=dims)
    return norm

def torch_nuc_norm(t):
    r"""Compute the Frobenius-norm across all entries in 
        tensor `t` along all dimensions.
    """
    dims = list(range(t.dim()))
    norm = torch.norm(t.cpu(), p='nuc', dim=dims)
    return norm

def operatornorm_bigH(kern, n, m, order):
    """For input images of size nxm, this finds the matrix 
        operator norm of the matrix associated to the the 
        2D convolution. Order can be 'nuc', 'fro' or 2.

        NOTE: Significantly slower than fourier_operatornorm,
            but is used for checking validity. Also, this 
            method assumes zero-padding whereas the Fourier
            method assumes periodic boundary condiditons.
    Args:
        - kern (torch.Tensor): Convolutional filter matrix.
        - n,m (int): dimension of the (mock) input image.
        - order (string): String describing the operator norm.

    Returns: norm of filter (float).
    """
    kern = kern.cpu().detach().numpy()
    k1, k2 = kern.shape
    filtmat = np.zeros(shape=(n+k1-1,m+k2-1),dtype=np.float32)
    filtmat[-k1:,:k2] = kern
    
    toeplitz_mats = []
    for i in range(filtmat.shape[0]-1, -1, -1):
        c = filtmat[i,:]
        r = np.concatenate([[c[0]], np.zeros(m-1)])
        toeplitz_m = scipy.linalg.toeplitz(c,r)
        toeplitz_mats.append(toeplitz_m)

    c = range(1, filtmat.shape[0]+1)
    r = np.concatenate([[c[0]], np.zeros(n-1, dtype=int)])
    doubly_indices = scipy.linalg.toeplitz(c, r)
    toeplitz_shape = toeplitz_mats[0].shape
    h = toeplitz_shape[0]*doubly_indices.shape[0]
    w = toeplitz_shape[1]*doubly_indices.shape[1]
    doubly_blocked_shape = [h, w]
    doubly_blocked = np.zeros(doubly_blocked_shape)

    b_h, b_w = toeplitz_shape
    for i in range(doubly_indices.shape[0]):
        for j in range(doubly_indices.shape[1]):
            start_i = i * b_h
            start_j = j * b_w
            end_i = start_i + b_h
            end_j = start_j + b_w
            doubly_blocked[start_i: end_i, start_j:end_j] = toeplitz_mats[doubly_indices[i,j]-1]
    if order == 'rank':
        normbigH = np.linalg.matrix_rank(doubly_blocked)
    else:
        normbigH = np.linalg.norm(doubly_blocked, order)
    return normbigH

def compute_BigH_norms(t, M, orde):
    r"""Computes the norm of a convolutional layer as interpreted
        as a matrix operator. Uses the slower method that actually
        builds the matrix. Is used to check the validity.

    Args:
        t (torch.Tensor): tensor representing the parameter to prune
        M (int): Dimension of (mock) input image.
        orde (string, int): Order of the norm. Either a string 'nuc'
            or 'fro', or an integer.

    Returns:
        norm (list(float)): List of list of norms for all channels,
            and all inputs for the tensor.
    """
    if t.size()[-1] == 3 and t.size()[-2]==3:
        tnorms = []
        for x in range(t.size()[0]):
            normsx = []
            for y in range(t[x].size()[0]):
                kernel = t[x][y]
                knorm = operatornorm_bigH(kernel, M, M, orde)
                normsx.append(knorm)
            tnorms.append(normsx)
        tnorms = torch.Tensor(tnorms)
    else:
        raise Exception("Asked to compute norm of non-convolutional layer!")
    return tnorms


######################################################################
### Utility functions that are not used, but useful for inspection ###
######################################################################

def build_bigH(kern, n, m, order):
    r"""For input images of size nxm, this finds the matrix 
        associated to the 2D convolution and computes the 
        norm. Order can be 'nuc', 'fro' or 2.
    """
    kern = kern.cpu().detach().numpy()
    k1, k2 = kern.shape
    filtmat = np.zeros(shape=(n+k1-1,m+k2-1),dtype=np.float32)
    filtmat[-k1:,:k2] = kern

    toeplitz_mats = []
    for i in range(filtmat.shape[0]-1, -1, -1):
        c = filtmat[i,:]
        r = np.concatenate([[c[0]], np.zeros(m-1)])
        toeplitz_m = scipy.linalg.toeplitz(c,r)
        toeplitz_mats.append(toeplitz_m)

    c = range(1, filtmat.shape[0]+1)
    r = np.concatenate([[c[0]], np.zeros(n-1, dtype=int)])
    doubly_indices = scipy.linalg.toeplitz(c, r)
    toeplitz_shape = toeplitz_mats[0].shape
    h = toeplitz_shape[0]*doubly_indices.shape[0]
    w = toeplitz_shape[1]*doubly_indices.shape[1]
    doubly_blocked_shape = [h, w]
    doubly_blocked = np.zeros(doubly_blocked_shape)

    b_h, b_w = toeplitz_shape
    for i in range(doubly_indices.shape[0]):
        for j in range(doubly_indices.shape[1]):
            start_i = i * b_h
            start_j = j * b_w
            end_i = start_i + b_h
            end_j = start_j + b_w
            doubly_blocked[start_i: end_i, start_j:end_j] = toeplitz_mats[doubly_indices[i,j]-1]
    return doubly_blocked

import torch
import msd_pytorch as mp
from torch.utils.data import DataLoader
import numpy as np
import torch.nn as nn
import torch.nn.utils.prune as prune
import scipy.linalg

def operator_norm_fourier_torch(w, stride, n):
    r"""
    NOTE: ONLY WORKS WITH PyTorch >= 1.9.0, the numpy version is equivalent
    Args:
      w (torch.tensor): convolutional filter.
      stride (int): convolution stride.
      n (int): image size (n x n image).
    """
    if stride == 1:
        return torch.fft.fftn(w, s=(n, n), norm="backward").abs().max()
    ws = [w[i::stride, j::stride] for i in range(stride) for j in range(stride)]
    spectrum = torch.stack([torch.fft.fftn(w, s=(n, n), norm="backward") for w in ws])
    return spectrum.abs().norm(p=2, dim=0).abs().max()

def operator_norm_fourier(w, stride, n):
    r"""
    Args:
      w (np.array): convolutional filter.
      stride (int): convolution stride.
      n (int): image size (n x n image).
    """
    if stride == 1:
        return np.abs(np.fft.fftn(w, s=(n, n), norm="backward")).max()
    ws = [w[i::stride, j::stride] for i in range(stride) for j in range(stride)]
    spectrum = np.stack([np.fft.fftn(w, s=(n, n), norm="backward") for w in ws])
    Svals = np.linalg.norm(np.abs(spectrum), axis=0) # is 2 norm by default for vectors
    specnorm = Svals.max()
    return specnorm

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

def compute_norm_stride2(mat, N, order):
    if mat.shape[0] != mat.shape[1]:
        raise Exception("Something gone terribly wrong here")
    if mat.shape[0] == 2:
        return np.sqrt(np.abs(mat**2).sum())
    siz = mat.shape[0]//2 + (mat.shape[0] % 2)
    f1 = np.zeros(shape=(siz,siz))
    f2 = np.zeros(shape=(siz,siz))
    f3 = np.zeros(shape=(siz,siz))
    f4 = np.zeros(shape=(siz,siz))
    f1 += mat[0::2,0::2]
    f2mat = mat[0::2,1::2]
    f2[:f2mat.shape[0], :f2mat.shape[1]] += f2mat
    f3mat = mat[1::2,0::2]
    f3[:f3mat.shape[0], :f3mat.shape[1]] += f3mat
    f4mat = mat[1::2,1::2]
    f4[:f4mat.shape[0], :f4mat.shape[1]] += f4mat

    f1fft = np.fft.fft2(f1, [N//2, N//2], axes=[0, 1])
    f2fft = np.fft.fft2(f2, [N//2, N//2], axes=[0, 1])
    f3fft = np.fft.fft2(f3, [N//2, N//2], axes=[0, 1])
    f4fft = np.fft.fft2(f4, [N//2, N//2], axes=[0, 1])

    #Fast numpy one-liner
    P = np.array([f1fft, f2fft, f3fft, f4fft])
    Svals = np.linalg.norm(P, axis=0) # is 2 norm by default for vectors
    if order == 'max':
        return np.max(Svals)
    else:
        if order == 'nuc':
            return np.sum(Svals)
        elif order == 'fro':
            frob = np.sum(Svals**2)
            return np.sqrt(frob)
        else:
            raise Exception("Not implemented order")

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
    doubly_blocked = build_bigH(kern, n, m)
    if order == 'rank':
        normbigH = np.linalg.matrix_rank(doubly_blocked)
    else:
        normbigH = np.linalg.norm(doubly_blocked, order)
    return normbigH


def compute_FourierSVD_norms(t, M, orde, strides):
    r"""Computes the norm of a convolutional layer as interpreted
        as a matrix operator.

    Args:
        t (torch.Tensor): tensor representing the parameter to prune
        M (int): Dimension of (mock) input image.
        orde (string, int): Order of the norm. Either a string 'nuc'
            or 'fro', or an integer.
        strides (tuple<int>): tuple of strides (x,y) for both
            convolution dimensions.

    Returns:
        norm (list(float)): List of list of norms for all channels,
            and all inputs for the tensor.
    """
    if t.size()[-1] >= 1  and t.size()[-2] >= 1:
        if strides == (1,1):
            tnorms = []
            for x in range(t.size()[0]):
                normsx = []
                for y in range(t[x].size()[0]):
                    kernel = t[x][y]
                    knorm = fourier_operatornorm(kernel, M, orde)
                    normsx.append(knorm)
                tnorms.append(normsx)
            tnorms = torch.Tensor(tnorms)
        elif strides == (2,2):
            tnorms = []
            tnumpy = t.detach().cpu().numpy()
            for x in range(tnumpy.shape[0]):
                normsx = []
                for y in range(tnumpy.shape[1]):
                    kernel = tnumpy[x][y]
                    knorm = compute_norm_stride2(kernel, M, orde)
                    normsx.append(knorm)
                tnorms.append(normsx)
            tnorms = torch.Tensor(tnorms)
        else:
            raise Exception("Not implemented for this stride size")
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


######################################################################
### Utility functions that are not used, but useful for inspection ###
######################################################################

def power_method(mat, n, iters=20):
    x1 = np.ones((n,n))
    for k in range(iters):
        x2 = scipy.signal.convolve2d(mat, x1)
        max_val = max(abs(np.amax(x2)), abs(np.amin(x2)))
        x1 = x2/max_val
    return max_val

def build_bigH(kern, n, m):
    r"""For input images of size nxm, this finds the matrix 
        associated to the 2D convolution and computes the 
        norm. Order can be 'nuc', 'fro' or 2.
    """
    k1, k2 = kern.shape
    filtmat = np.zeros(shape=(n+k1-1,m+k2-1),dtype=np.float32)
    ## NOTE: scipy implements with reversed kernel!!! So uncomment if
    # comparing to scipy.convolve
    #filtmat[-k1:,:k2] = kern
    filtmat[-k1:,:k2] = np.flip(kern, 0)

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

def build_strided_bigH(kern, N, strid):
    # Build strided bigH matrix, only for square images
    pad = kern.shape[0] // 2

    bigH = build_bigH(kern, N, N)
    padN = N + 2*pad
    stridN = N//strid
    if N % strid != 0:
        stridN += 1
    bigHstrid = np.zeros(shape=(stridN*stridN, bigH.shape[1]))
    # Only copy relevant rows
    it = 0
    if kern.shape[0] % 2 == 0:# For even convolutions
        pad -= 1
        padN -= 1
        for x in range(pad, pad + N):
            if (x - pad) % strid != 0:
                continue
            for y in range(pad, pad + N):
                if (y - pad) % strid != 0:
                    continue
                index = x*padN + y
                col = bigH[index,:]
                bigHstrid[it,:] = col
                it += 1
    else:
        for x in range(pad, pad + N):
            if (x - pad) % strid != 0:
                continue
            for y in range(pad, pad + N):
                if (y - pad) % strid != 0:
                    continue
                index = x*padN + y
                col = bigH[index,:]
                bigHstrid[it,:] = col
                it += 1
    return bigHstrid


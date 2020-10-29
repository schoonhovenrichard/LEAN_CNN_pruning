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

#import multiprocessing as mp
#import threading as thr

#TODO: Pruning on decay metric like in old/

def torch_norm(t, n, dim):
    r"""Compute the L_n-norm across all entries in tensor `t` along all dimension 
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
    r"""Compute the Frobenius-norm across all entries in tensor `t` along all dimensions.
    """
    dims = list(range(t.dim()))
    norm = torch.norm(t.cpu(), p='fro', dim=dims)
    return norm

def torch_nuc_norm(t):
    r"""Compute the Frobenius-norm across all entries in tensor `t` along all dimensions.
    """
    dims = list(range(t.dim()))
    norm = torch.norm(t.cpu(), p='nuc', dim=dims)
    return norm

def build_bigH(kern, n, m, order):
    """For input images of size nxm, this finds the matrix multiplication variant of
    the 2D convolution and computes the norm. Order can be 'nuc', 'fro' or 2.
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

def power_operator_norm(A, num_iter=30):
    x = torch.randn(A.T.shape)
    for i in range(num_iter):
        x = A.T@ A @ x.numpy()
        x = torch.Tensor(x)
        x /= torch.norm(x)  # L2 vector-norm

    norm_ATA = (torch.norm(torch.Tensor(A.T @ A @ x.numpy())) / torch.norm(x)).item()
    return math.sqrt(norm_ATA)

def fourier_operatornorm(kern, n, order):
    r"""Computes the FFT of the filter matrix to efficiently determine the SVD.
    From the paper: The Singular Values of Convolutional Layers, 
    by Hanie Sedghi, Vineet Gupta, Philip M. Long
    """
    filt = kern.detach().cpu().numpy()
    if order == 'max':
        fourier_coeff = np.fft.fft2(filt, [n,n], axes=[0, 1])
        return np.max(np.abs(fourier_coeff))
    else:
        S = np.sort(np.absolute(np.fft.fft2(filt, [n, n]).flatten()))[::-1]
        if order == 'nuc':
            return np.sum(S)
        elif order == 'fro':
            frob = np.sum(S**2)
            return np.sqrt(frob)
        else:
            raise Exception("Not implemented order")

def operatornorm_bigH(kern, n, m, order):
    """For input images of size nxm, this finds the matrix multiplication variant of
    the 2D convolution and computes the norm. Order can be 'nuc', 'fro' or 2.
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

def compute_FourierSVD_norms(t, M, orde):
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

def compute_BigH_norms(t, M, orde):
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

def compute_Ln_norms_conv(t, N):
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

class CustomLnMethod(prune.BasePruningMethod):
    """
    Structured Ln pruning on each 3x3 convolution
    """
    PRUNING_TYPE='structured'

    def __init__(self, amount, n, dim):
        # Check range of validity of pruning amount
        prune._validate_pruning_amount_init(amount)
        self.amount = amount
        self.n = n
        self.dim = dim

    @classmethod
    def apply(cls, module, name, amount, n, dim):
        return super(CustomLnMethod, cls).apply(module, name, amount=amount, n=n, dim=dim)

    def compute_mask(self, t, default_mask):
        if t.size()[-1] == 3 and t.size()[-2]==3:
            tensor_size = t.shape[self.dim]
            nparams_toprune = int(round(self.amount * tensor_size))
            nparams_tokeep = tensor_size - nparams_toprune
            if nparams_toprune == 0:  # k=0 not supported by torch.kthvalue
                mask = default_mask
                return mask
            if self.n == 'nuc' or self.n == 'fro':
                tensor_size = int(t.numel()/9)
                nparams_toprune = int(round(self.amount * tensor_size))
                nparams_tokeep = tensor_size - nparams_toprune
                all_norms = []
                tnorms = []
                for x in range(t.size()[0]):
                    normsx = []
                    for y in range(t[x].size()[0]):
                        kernel = t[x][y]
                        if self.n == 'nuc':
                            knorm = torch_nuc_norm(kernel)
                        elif self.n == 'fro':
                            knorm = torch_fro_norm(kernel)
                        else:
                            raise Exception("Order not implemented")
                        if knorm > 0:
                            all_norms.append(knorm)
                        normsx.append(knorm)
                    tnorms.append(normsx)
                threshold = np.percentile(np.array(all_norms), 100*self.amount)

                mask = torch.ones_like(t)
                for i in range(len(tnorms)):
                    for j in range(len(tnorms[i])):
                        if tnorms[i][j] < threshold:
                            mask[i][j] = torch.zeros_like(t[i][j])
                mask *= default_mask.to(dtype=mask.dtype)
            else:
                norm = torch_norm(t, self.n, self.dim)
                topk = torch.topk(norm, k=nparams_tokeep, largest=True,)
                def make_mask(t, dim, indices):
                    mask = torch.zeros_like(t)
                    slc = [slice(None)] * len(t.shape)
                    slc[dim] = indices
                    mask[slc] = 1
                    return mask
                mask = make_mask(t, self.dim, topk.indices)
                mask *= default_mask.to(dtype=mask.dtype)
        else:
            mask = default_mask
        return mask

def customln_structured(module, name, amount, n, dim=1):
    CustomLnMethod.apply(module, name, amount, n, dim)
    return module

class BigHPruningMethod(prune.BasePruningMethod):
    PRUNING_TYPE='structured'

    def __init__(self, amount, n, norm, dim):
        # Check range of validity of pruning amount
        prune._validate_pruning_amount_init(amount)
        self.amount = amount
        self.dim = dim
        self.n = n
        self.norm = norm

    @classmethod
    def apply(cls, module, name, amount, n, norm, dim):
        return super(BigHPruningMethod, cls).apply(module, name, amount=amount, n=n, norm=norm, dim=dim)

    def operatornorm_bigH_thr(self, kernel, y, outQ): #Deprecated
        knorm = operatornorm_bigH(kernel, self.n, self.n, self.norm)
        outQ.put([y,knorm])

    def compute_mask(self, t, default_mask):
        if t.size()[-1] == 3 and t.size()[-2]==3:
            tensor_size = int(t.numel()/9)
            nparams_toprune = int(round(self.amount * tensor_size))
            nparams_tokeep = tensor_size - nparams_toprune
            if nparams_toprune == 0:  # k=0 not supported by torch.kthvalue
                mask = default_mask
                return mask
            else:
                all_norms = []
                tnorms = []
                for x in range(t.size()[0]):
                    normsx = []
                    for y in range(t[x].size()[0]):
                        kernel = t[x][y]
                        knorm = operatornorm_bigH(kernel, self.n, self.n, self.norm)
                        if knorm > 0:
                            all_norms.append(knorm)
                        normsx.append(knorm)
                    tnorms.append(normsx)
                threshold = np.percentile(np.array(all_norms), 100*self.amount)
            
                tnorms = torch.Tensor(tnorms)
                mask = torch.ones_like(t)
                for i in range(len(tnorms)):
                    for j in range(len(tnorms[i])):
                        if tnorms[i][j] < threshold:
                            mask[i][j] = torch.zeros_like(t[i][j])
                mask *= default_mask.to(dtype=mask.dtype)
        else:
            mask = default_mask
        return mask

def BigH_structured(module, name, amount, n, norm, dim=1):
    BigHPruningMethod.apply(module, name, amount, n, norm, dim)
    return module

class FourierSVDPruningMethod(prune.BasePruningMethod):
    PRUNING_TYPE='structured'

    def __init__(self, amount, n, norm, dim):
        # Check range of validity of pruning amount
        prune._validate_pruning_amount_init(amount)
        self.amount = amount
        self.dim = dim
        self.n = n
        self.norm = norm

    @classmethod
    def apply(cls, module, name, amount, n, norm, dim):
        return super(FourierSVDPruningMethod, cls).apply(module, name, amount=amount, n=n, norm=norm, dim=dim)

    def compute_mask(self, t, default_mask):
        if t.size()[-1] == 3 and t.size()[-2]==3:
            tensor_size = int(t.numel()/9)
            nparams_toprune = int(round(self.amount * tensor_size))
            nparams_tokeep = tensor_size - nparams_toprune
            if nparams_toprune == 0:  # k=0 not supported by torch.kthvalue
                mask = default_mask
                return mask
            else:
                all_norms = []
                tnorms = []
                for x in range(t.size()[0]):
                    normsx = []
                    for y in range(t[x].size()[0]):
                        kernel = t[x][y]
                        knorm = fourier_operatornorm(kernel, self.n, self.norm)
                        if knorm > 0:
                            all_norms.append(knorm)
                        normsx.append(knorm)
                    tnorms.append(normsx)
                threshold = np.percentile(np.array(all_norms), 100*self.amount)
            
                tnorms = torch.Tensor(tnorms)
                mask = torch.ones_like(t)
                for i in range(len(tnorms)):
                    for j in range(len(tnorms[i])):
                        if tnorms[i][j] < threshold:
                            mask[i][j] = torch.zeros_like(t[i][j])
                mask *= default_mask.to(dtype=mask.dtype)
        else:
            mask = default_mask
        return mask

def FourierSVD_structured(module, name, amount, n, norm, dim=1):
    FourierSVDPruningMethod.apply(module, name, amount, n, norm, dim)
    return module

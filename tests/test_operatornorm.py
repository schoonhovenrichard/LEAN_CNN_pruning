import scipy.signal
import numpy as np
import random
import pytest

def test_operator_norm_power(tries=20, accept_error=0.05):
    """Test computation of operator norm of convolution
        against the power method, which uses explicit
        repeated application of the convolution operator.
        
    We check that:
    - The norms are non-negative
    - The error between the two norms, which will exist due
        to the inaccuracy of the power method for this
        number of iterations, is less than 5%.
    """
    N = 32
    ks = [1,3,5,7,9]

    # Test against power method
    percerror = 0.0
    for r in range(10):
        k = random.choice(ks)
        matrix = generate_random_filter(3)
        fournorm = fourier_operatornorm(matrix, N)
        powernorm = power_method(matrix, N)
        assert fournorm >= 0
        assert powernorm >= 0
        percerror += np.abs(fournorm - powernorm)/float(fournorm)
    percerror = percerror/float(tries)

    # There will be discrepancy because power method is not
    # accurate for this many iterations
    assert percerror <= 0.05,"Unit test failed for operator norm!"

def test_operator_norm_bigH(tries=100, accept_error=0.005):
    """Test computation of operator norm of convolution
        against explicitly computing the matrix associated
        with the convolution.
        
    We check that:
    - The norms are non-negative
    - The error between the two norms, which will exist due
        to reflective vs. zero-padding assumptions, is less
        than 0.5%.
    """
    N = 32
    ks = [1,3,5,7,9]

    # Test against BigH method
    percerror = 0.0
    for r in range(tries):
        k = random.choice(ks)
        matrix = generate_random_filter(3)
        fournorm = fourier_operatornorm(matrix, N)
        bighnorm = operatornorm_bigH(matrix, N, N, 2)
        assert fournorm >= 0
        assert bighnorm >= 0
        percerror += np.abs(fournorm - bighnorm)/float(fournorm)
    percerror = percerror/float(tries)

    # There will be discrepancy because of padding
    assert percerror <= accept_error,"Unit test failed for operator norm!"

def generate_random_filter(filt_size):
    sampl = np.random.uniform(low=-10.0, high=10.0, size=(filt_size*filt_size,))
    filt = sampl.reshape((filt_size, filt_size))
    return filt

def fourier_operatornorm(filt, n):
    fourier_coeff = np.fft.fft2(filt, [n,n], axes=[0, 1])
    return np.max(np.abs(fourier_coeff))
    #S = np.sort(np.absolute(np.fft.fft2(filt, [n, n]).flatten()))[::-1]
    #S = np.linalg.svd(fourier_coeff, compute_uv=False, full_matrices=False)
    #return np.max(S)

def operatornorm_bigH(kern, n, m, order):
    """For input images of size nxm, this finds the matrix multiplication variant of
    the 2D convolution and computes the norm. Order can be 'nuc', 'fro' or 2.
    """
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

def power_method(mat, n, iters=40):
    x1 = np.ones((n,n))
    for k in range(iters):
        x2 = scipy.signal.convolve2d(mat, x1)
        max_val = max(abs(np.amax(x2)), abs(np.amin(x2)))
        x1 = x2/max_val
    return max_val


if __name__ == '__main__':
    run_operator_norm_test()
    
    N = 32
    #mat = np.array([[ 0.6517,  0.7637,  0.3560],
    #        [ 1.8497, -3.3352, -0.0926],
    #        [ 1.0303, -1.1776,  0.0818]])
    #print(fourier_operatornorm(mat, N))
    #print(operatornorm_bigH(mat, N, N, 2))
    #print(power_method(mat, N))

    test_power_method = False
    if test_power_method:
        ks = [1,3,5,7,9]
        percerror = 0.0
        tries = 100
        for r in range(tries):
            k = random.choice(ks)
            matrix = generate_random_filter(3)
            fournorm = fourier_operatornorm(matrix, N)
            powernorm = power_method(matrix, N)
            percerror += np.abs(fournorm - powernorm)/float(fournorm)
        percerror = percerror/float(tries)
        print(percerror)

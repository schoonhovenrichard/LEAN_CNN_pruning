import scipy.signal
import numpy as np
import random
import pytest
import norm_calculation_methods as leannorm
import torch

def generate_random_filter(filt_size):
    sampl = np.random.uniform(low=-10.0, high=10.0, size=(filt_size*filt_size,))
    filt = sampl.reshape((filt_size, filt_size))
    return filt

def test_operator_norm_bigH(tries=20, accept_error=0.01):
    """Test computation of operator norm of convolution
        against explicitly computing the matrix associated
        with the convolution.
        
    We check that:
    - The norms are non-negative
    - The error between the two norms, which will exist due
        to reflective vs. zero-padding assumptions, is less
        than 0.5%.
    """
    N = 48
    ks = [1,3,5,7,9]

    # Test against BigH method
    percerror = 0.0
    for r in range(tries):
        k = random.choice(ks)
        matrix = generate_random_filter(k)
        kernel = torch.Tensor(matrix)
        fournorm = leannorm.fourier_operatornorm(kernel, N, 'max')
        bighnorm = leannorm.operatornorm_bigH(kernel, N, N, 2)
        assert fournorm >= 0
        assert bighnorm >= 0
        percerror += np.abs(fournorm - bighnorm)/float(fournorm)
    percerror = percerror/float(tries)

    # There will be discrepancy because of padding
    assert percerror <= accept_error,"Unit test failed for operator norm!"

def test_bigH_correctness1(tries=100, accept_error=0.005):
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
    err = 0.0
    for r in range(tries):
        k = random.choice(ks)
        matrix = generate_random_filter(k)
        img = np.random.rand(N,N)

        # Control array
        k1 = matrix.shape[0]
        k2 = matrix.shape[1]
        padx = k1//2
        pady = k2//2
        control_arr = scipy.signal.convolve2d(matrix, img, boundary='fill')

        # Check if the BigH multiplication does the same thing.
        bigH = leannorm.build_bigH(matrix, N, N)
        # Perform matrix multiplication and check the result
        flat_arr = bigH.dot(img.flatten())
        Q = int(np.sqrt(flat_arr.shape[0]))
        result_arr = flat_arr.reshape((Q,Q))
        err += np.abs(result_arr - control_arr).sum()
    err = err/float(tries)

    # There will be discrepancy because of padding
    assert err <= accept_error,"Unit test failed for operator norm!"

def test_bigH_correctness2(tries=100, accept_error=0.005):
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
    err = 0.0
    for r in range(tries):
        k = random.choice(ks)
        matrix = generate_random_filter(k)
        img = np.random.rand(N,N)

        # Control array
        k1 = matrix.shape[0]
        k2 = matrix.shape[1]
        padx = k1//2
        pady = k2//2

        padded_img = np.pad(img, ((padx, padx), (pady, pady)), 'constant', constant_values=0)
        control_arr = convolve2D(padded_img, matrix)[:img.shape[0],:img.shape[1]]

        # Check if the BigH multiplication does the same thing.
        mat = matrix
        bigH = leannorm.build_bigH(mat, N, N)
        # Perform matrix multiplication and check the result
        flat_arr = bigH.dot(img.flatten())
        Q = int(np.sqrt(flat_arr.shape[0]))

        #NOTE This matrix multiplication "enlarges the image to include
        # the padded rows and columns. If we pad with k//2+1 for the 
        # control we get the same result. So we either pad with k//2+1 
        # for the control, or we remove the excess padded rows
        # and columns from the bigH result.
        minx = padx
        miny = pady
        if k1 % 2 == 0:
            minx -= 1
        if k2 % 2 == 0:
            miny -= 1
        minx = max(minx, 0)
        miny = max(miny, 0)
        if padx > 0 and pady > 0:
            result_arr = flat_arr.reshape((Q,Q))[minx:-padx, miny:-pady]
        else:
            result_arr = flat_arr.reshape((Q,Q))
        err += np.abs(result_arr - control_arr).sum()
    err = err/float(tries)

    # There will be discrepancy because of padding
    assert err <= accept_error,"Unit test failed for operator norm!"

def test_operator_norm_bigH_stride1(tries=30, accept_error=0.01):
    """Test computation of operator norm of convolution
        against explicitly computing the matrix associated
        with the convolution when using strided convolutions.
        
    We check that:
    - The norms are non-negative
    - The error between the two norms, which will exist due
        to reflective vs. zero-padding assumptions, is less
        than 0.5%.
    """
    N = 48
    ks = [1,3,5,7,9]
    strid = 2

    # Test against BigH method
    percerror = 0.0
    for r in range(tries):
        k = random.choice(ks)
        matrix = generate_random_filter(k)
        # 'max' is maximum singular value, i.e., spectral norm
        fournorm = leannorm.compute_norm_stride2(matrix, N, 'max')

        bigHstrid = leannorm.build_strided_bigH(matrix, N, strid)
        bighnorm = np.linalg.norm(bigHstrid, 2)#Use 2 norm for spectral norm
        assert fournorm >= 0
        assert bighnorm >= 0
        percerror += np.abs(fournorm - bighnorm)/float(fournorm)
    percerror = percerror/float(tries)

    # There will be discrepancy because of padding
    assert percerror <= accept_error,"Unit test failed for operator norm!"

def test_operator_norm_bigH_stride2(tries=30, accept_error=0.01):
    """Test computation of operator norm of convolution
        against explicitly computing the matrix associated
        with the convolution when using strided convolutions.
        
    We check that:
    - The norms are non-negative
    - The error between the two norms, which will exist due
        to reflective vs. zero-padding assumptions, is less
        than 0.5%.
    """
    N = 48
    ks = [1,3,5,7,9]
    strid = 2

    # Test against BigH method
    percerror = 0.0
    for r in range(tries):
        k = random.choice(ks)
        matrix = generate_random_filter(k)
        fournorm = leannorm.operator_norm_fourier(matrix, strid, N)

        bigHstrid = leannorm.build_strided_bigH(matrix, N, strid)
        bighnorm = np.linalg.norm(bigHstrid, 2)#Use 2 norm for spectral norm
        assert fournorm >= 0
        assert bighnorm >= 0
        percerror += np.abs(fournorm - bighnorm)/float(fournorm)
    percerror = percerror/float(tries)

    # There will be discrepancy because of padding
    assert percerror <= accept_error,"Unit test failed for operator norm!"

def ttest_operator_norm_bigH_stride3(tries=10, accept_error=0.01):
    """Test computation of operator norm of convolution
        against explicitly computing the matrix associated
        with the convolution when using strided convolutions.
        
    We check that:
    - The norms are non-negative
    - The error between the two norms, which will exist due
        to reflective vs. zero-padding assumptions, is less
        than 0.5%.
    """
    N = 48
    ks = [1,3,5,7,9]
    strid = 2

    # Test against BigH method
    percerror = 0.0
    for r in range(tries):
        k = random.choice(ks)
        matrix = generate_random_filter(k)
        bigHstrid = leannorm.build_strided_bigH(matrix, N, strid)
        bighnorm = np.linalg.norm(bigHstrid, 2)#Use 2 norm for spectral norm

        kernel = torch.Tensor(matrix)
        fournorm = leannorm.operator_norm_fourier_torch(kernel, strid, N)
        assert fournorm >= 0
        assert bighnorm >= 0
        percerror += np.abs(fournorm - bighnorm)/float(fournorm)
    percerror = percerror/float(tries)

    # There will be discrepancy because of padding
    assert percerror <= accept_error,"Unit test failed for operator norm!"

def test_operator_norm_power(tries=10, accept_error=0.1):
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
    for r in range(tries):
        k = random.choice(ks)
        matrix = generate_random_filter(k)
        mattorch = torch.Tensor(matrix)
        # 'max' is maximum singular value, i.e., spectral norm
        fournorm = leannorm.fourier_operatornorm(mattorch, N, 'max')
        powernorm = leannorm.power_method(matrix, N)
        assert fournorm >= 0
        assert powernorm >= 0
        percerror += np.abs(fournorm - powernorm)/float(fournorm)
    percerror = percerror/float(tries)

    # There will be discrepancy because power method is not
    # accurate for this many iterations
    assert percerror <= 0.05,"Unit test failed for operator norm!"


def convolve2D(image, kernel, padding=0, strides=1):
    r"""
    This method is adapted from https://medium.com/analytics-vidhya/2d-convolution-using-python-numpy-43442ff5f381
    
    We need it because scipy does not implement striding.
    The results are the same if you add padding to the input image.
    """
    # Cross Correlation
    kernel = np.flipud(np.fliplr(kernel))

    # Gather Shapes of Kernel + Image + Padding
    xKernShape = kernel.shape[0]
    yKernShape = kernel.shape[1]
    xImgShape = image.shape[0]
    yImgShape = image.shape[1]

    # Shape of Output Convolution
    #NOTE: In the case of even kernel sizes, this means that the convolution is defined as
    # starting in the upper left. E.g. for 2x2, the bottom right convolution element is
    # multiplied with top left pixel of the image, and the rest is zeroes.
    xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
    yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)

    output = np.zeros((xOutput, yOutput))

    # Apply Equal Padding to All Sides
    if padding != 0:
        imagePadded = np.zeros((image.shape[0] + padding*2, image.shape[1] + padding*2))
        imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
    else:
        imagePadded = image

    # Iterate through image
    for y in range(image.shape[1]):
        # Exit Convolution
        if y > image.shape[1] - yKernShape:
            break
        # Only Convolve if y has gone down by the specified Strides
        if y % strides == 0:
            for x in range(image.shape[0]):
                # Go to next row once kernel is out of bounds
                if x > image.shape[0] - xKernShape:
                    break
                # Only Convolve if x has moved by the specified Strides
                if x % strides == 0:
                    output[x//strides, y//strides] = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum()
    return output


if __name__ == '__main__':
    from timeit import default_timer as timer

    test_bigH_correctness1()
    test_bigH_correctness2()
    test_operator_norm_bigH()
    test_operator_norm_bigH_stride1()
    test_operator_norm_bigH_stride2()
    test_operator_norm_power()
    #ttest_operator_norm_bigH_stride3()# Does not work with old pytorch

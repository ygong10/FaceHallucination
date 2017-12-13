import numpy as np
from scipy.misc import imread
from scipy.ndimage.filters import convolve
from skimage.transform import pyramid_gaussian, pyramid_laplacian

# Sobel filters
first_sobel_horiz = np.array([
    [1, 0, -1],
    [2, 0, -2],
    [1, 0, -1]
])

first_sobel_vert = np.array([
    [1, 2, 1],
    [0, 0, 0],
    [-1, -2, -1]
])

# https://dsp.stackexchange.com/a/10622
second_sobel_horiz = np.array([
    [1, -2, 1],
    [2, -4, 2],
    [1, -2, 1]
])

second_sobel_vert = np.array([
    [1, 2, 1],
    [-2, -4, -2],
    [1, 2, 1]
])

def W(m, n, p, q, k):
    if m * 2**k <= p < (m+1) * 2**k and n * 2**k <= q < (n+1) * 2**k:
        return 1 / 2 ** (2*k)
    else:
        return 0

def map_formulation(G_0, G_k, height, width, k=2, covariance=1.0):
    # calculates -ln Pr[G_k | G_0]
    # G_0 - predicted first image of Gaussian pyramid
    # G_k - image at the kth level of the Gaussian pyramid
    return 1 / (2*covariance) * sum(
        sum(G_k[m, n] - sum(
            sum(
                W(m, n, p, q, k) * G_0[p, q]
                for q in range(width*2**k))
            for p in range(height*2**k))
            for n in range(width))
        for m in range(height))

def gradient_prior(H0_G0, V0_G0, H0_I, V0_I, height, width, k=2, covariance=1.0):
    # calculates -ln Pr[G_0]
    # H0_G0 - horizontal derivative of G_0
    # V0_G0 - vertical derivative of G_0
    # H0_I - predicted horizontal derivative of G_0
    # V0_I - predicted vertical derivative of G_0
    return 1 / (2*covariance) * sum(
        sum(
            sum((H0_G0[m, n] - H0_I[m, n]) ** 2
            for n in range(width))
            for m in range(height))) + 1 / (2*covariance) * sum(
        sum(
            sum((V0_G0[m, n] - V0_I[m, n]) ** 2
            for n in range(width))
            for m in range(height)))

def F(g_layer, l_layer):
    # create feature vector:
    # (laplacian, horizontal derivative, vertical derivative, 2nd h.d., 2nd v.d.)
    return (
        l_layer, convolve(g_layer, first_sobel_horiz), convolve(g_layer, first_sobel_vert),
        convolve(g_layer, second_sobel_horiz), convolve(g_layer, second_sobel_vert)
    )

def PS(im, depth):
    g_pyramid = pyramid_gaussian(im, max_layer=depth)
    l_pyramid = pyramid_gaussian(im, max_layer=depth)
    return [F(g_layer, l_layer) for g_layer, l_layer in zip(g_pyramid, l_pyramid)]

def predict_high_res(training_ims, low_res_im, k=2, N=4):
    low_res_fvs = PS(low_res_im, N-k)
    def PS_error(im):
        fvs = PS(im, N)[k:]
        weights = np.array([1, 0.5, 0.5, 0.5, 0.5])
        error = 0
        for i in range(N-k):
            error += sum(weights * [np.linalg.norm(lr_fv-fv, ord=2) for lr_fv, fv in zip(low_res_fvs[i], fvs[i])])
            weights /= 2
        return error

    high_res = np.zeros((low_res_im.shape[0] * 2**k, low_res_im.shape[1] * 2**k))
    for m in range(low_res_im.shape[0] * 2**k):
        for n in range(low_res_im.shape[1] * 2**k):
            fv_errors = [PS_error(im) for im in training_ims]
            j = np.argmin(fv_errors)
            high_res[m, n] = training_ims[j][m, n]
    return high_res

im = imread('../data/CAFE-FACS-Orig/004_d2.pgm')
#g_pyramid = pyramid_gaussian(im, max_layer=2)
#g_pyramid = list(g_pyramid)
#l_pyramid = pyramid_laplacian(im, max_layer=2)
#l_pyramid = list(l_pyramid)

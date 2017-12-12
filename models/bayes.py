import numpy as np
from scipy.misc import imread
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
    return 1 / (2*covariance) * sum(
        sum(G_k[m, n] - sum(
            sum(
                W(m, n, p, q, k) * G_0[p, q]
                for q in range(width*2**k))
            for p in range(height*2**k))
            for n in range(width))
        for m in range(height))

def gradient_prior(H0_G0, V0_G0, predicted_H0_I, predicted_V0_I, height, width, k=2, covariance=1.0):
    # calculates -ln Pr[G_0]
    return 1 / (2*covariance) * sum(
        sum(
            sum((H0_G0[m, n] - predicted_H0_I[m, n]) ** 2
            for n in range(width))
            for m in range(height))) + 1 / (2*covariance) * sum(
        sum(
            sum((V0_G0[m, n] - predicted_V0_I[m, n]) ** 2
            for n in range(width))
            for m in range(height)))

im = imread('../data/CAFE-FACS-Orig/004_d2.pgm')
g_pyramid = pyramid_gaussian(im, max_layer=2)
g_pyramid = list(g_pyramid)
l_pyramid = pyramid_laplacian(im, max_layer=2)
l_pyramid = list(l_pyramid)

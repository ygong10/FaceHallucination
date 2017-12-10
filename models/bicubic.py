from scipy.misc import imresize

def bicubic(im):
    # im: 2-D numpy array (32x24)
    return imresize(im, (128, 96))

import numpy as np 
from PIL import Image
from copy import deepcopy
from scipy import ndimage
import matplotlib.pyplot as plt

# Converts a pgm image into a numpy ndarray
def convert_pgm_to_numpy(pgm_img):
  im = Image.open(pgm_img)
  return np.array(im)

# Downsample a numpy ndarray image by an integer factor
def down_sample(im, factor):
  width, height = im.shape
  if (factor > width or factor > height):
    raise ValueError("Subsample factor is bigger than image dimensions")
    
  im = im[::factor, ::factor]
  return im

# Normalize the pixel values from [0, 255] to [0, 1]
def normalize(im):
  return im / 255

# Generate a guassian pyramid and returns it as a list of images.
def generate_gaussian_pyramid(im, downsample_factor, sigma, levels):
  if (levels < 1):
    raise ValueError("Pyramid should have more than 1 level")

  width, height = im.shape
  im = deepcopy(normalize(im))
  gaussian_pyramid = [im]

  for level in range(1, levels + 1):
    im = deepcopy(im)
    im = ndimage.filters.gaussian_filter(im, sigma)
    im = down_sample(im, downsample_factor)
    gaussian_pyramid.append(im)
  
  return gaussian_pyramid

# im = convert_pgm_to_numpy("data/CAFE-FACS-Orig/004_d2.pgm")
# pyramid = generate_gaussian_pyramid(im, 2, 2, 3)
# plt.imshow(pyramid[0], cmap="gray")
# plt.show()

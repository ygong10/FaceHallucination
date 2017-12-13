import os, os.path
import numpy as np
from PIL import Image
from copy import deepcopy
from scipy import ndimage
import scipy
import matplotlib.pyplot as plt

# Creates low resolution training images along with corresponding high resolution training_images
def create_data_from_images(data_path, low_res_file_path, high_res_file_path, sigma, downsample_factor, levels):
  high_res_images = os.listdir(data_path)
  num_samples = len(high_res_images)

  sample_image = scipy.misc.imread(os.path.join(data_path, high_res_images[0]), mode='L')
  width, height = sample_image.shape
  sample_pyramid = generate_gaussian_pyramid(
      im=sample_image,
      downsample_factor=downsample_factor,
      sigma=sigma,
      levels=levels)

  low_res_width, low_res_height = sample_pyramid[-1].shape
  del sample_image
  del sample_pyramid
  high_res_data = np.zeros((width * height, num_samples))
  low_res_data = np.zeros((low_res_width * low_res_height, num_samples))

  for i in range(num_samples):
    high_res_image = scipy.misc.imread(os.path.join(data_path, high_res_images[i]), mode='L')
    pyramid = generate_gaussian_pyramid(
      im=high_res_image,
      downsample_factor=downsample_factor,
      sigma=sigma,
      levels=levels)
    high_res_image = pyramid[0].flatten()
    low_res_image = pyramid[-1].flatten()
    high_res_data[:, i] = high_res_image
    low_res_data[:, i] = low_res_image
    print ("Finished processing image " + str(i))

  np.save(low_res_file_path, low_res_data)
  print("Saving low resolution training data")
  np.save(high_res_file_path, high_res_data)
  print("Saving high resolution training data")
  print("Done")

# Converts an image into a numpy ndarray
# def convert_image_to_numpy(pgm_img):
#   im = scipy.misc.imread(pgm_img, mode='L')
#   return np.array(im)

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

  for level in range(1, levels):
    im = deepcopy(im)
    im = ndimage.filters.gaussian_filter(im, sigma)
    im = down_sample(im, downsample_factor)
    gaussian_pyramid.append(im)

  return gaussian_pyramid

im = scipy.misc.imread("aligned_data/ucsd_aligned_images/004_d2_aligned.png", mode='L')
print(im.shape)
pyramid = generate_gaussian_pyramid(im, 2, 2, 4)
plt.imshow(pyramid[0], cmap="gray")
plt.show()


IMAGE_PATH = 'aligned_data/ucsd_aligned_images/'
TRAINING_HIGH_RES_PATH = 'training/training_high_res'
TRAINING_LOW_RES_PATH = 'training/training_low_res'
SIGMA = 2 # Sigma for gaussian filter.
DOWNSAMPLE_FACTOR = 2 # Downsample factor for gaussian pyramid
PYRAMID_LEVELS = 4

# Run this to generate the training low resolution and high resolution images
# create_data_from_images(
#   data_path=IMAGE_PATH,
#   low_res_file_path=TRAINING_LOW_RES_PATH,
#   high_res_file_path=TRAINING_HIGH_RES_PATH,
#   sigma=SIGMA,
#   downsample_factor=DOWNSAMPLE_FACTOR,
#   levels=PYRAMID_LEVELS
# )

# How to recover a 1d low res (4th level of gaussian pyramid) from .npy back to a 2d low res
# test = np.load('training/training_low_res.npy')[:, 0]
# test = test.reshape(29, 38)
# plt.imshow(test, cmap="gray")
# plt.show()

#How to recover a 1d high res from .npy back to a 2d high res
# test = np.load('training/training_low_res.npy')[:, 0]
# test = test.reshape(454, 604)
# plt.imshow(test, cmap="gray")
# plt.show()

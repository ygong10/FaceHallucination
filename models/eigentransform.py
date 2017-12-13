import numpy as np
from PIL import Image
from copy import deepcopy
import scipy.linalg
import scipy.misc
import matplotlib.pyplot as plt

"""
  Face Hallucination using Eigentransformation

  Step 1. We take an input image and a training set of low resolution images and obtain a set of weights corresponding to low resolution images.

  Step 2. We take the low resolution weights and repeat Step 1 with a training set of the corresponding high resolution images to obtain a set of weights corresponding to high resolution images. Additionally, we further process the set of high resolution weights using a scaling factor, alpha and corresponding eigenvalue as constraints to bound the principal components.
"""
def load_data(low_res_file_path, high_res_file_path):
  low_res_images = np.load(low_res_file_path)
  high_res_images = np.load(high_res_file_path)

  return low_res_images, high_res_images

def get_eigenfaces_and_eigenvalues(images, mean_face):
  L = images - np.vstack(mean_face)
  R = np.matmul(L.T, L)
  eig_vals, eig_vecs = scipy.linalg.eig(R, left=True, right=False)
  # Converting the array of eigenvalues into a matrix of eigenvalues
  eig_vals_matrix = np.eye(eig_vals.shape[0]) * eig_vals
  # Get rid of complex values, I don't know why there's sometimes complex values.
  eigenfaces = np.real(np.matmul(np.matmul(L, eig_vecs), (np.linalg.inv(eig_vals_matrix)**(1/2))))
  return eigenfaces, eig_vals

def get_low_res_weights(images, input_image):
  m_l = np.mean(images, axis=1)
  E_l, eig_val_l = get_eigenfaces_and_eigenvalues(images, m_l)
  return get_weights(E_l, input_image, m_l)

def get_hallucinated_results(images, width, height, low_res_weights, alpha):
  m_h = np.mean(images, axis=1)
  E_h, eig_val_h = get_eigenfaces_and_eigenvalues(images, m_h)

  x_h = np.zeros((width * height,))
  for i in range(len(low_res_weights)):
    c_i = low_res_weights[i]
    x_h += (c_i * images[:, i]) + m_h

  w_h = get_weights(E_h, x_h, m_h)

  for i in range(len(w_h)):
    constraint = (alpha * (np.real(eig_val_h[i]**(1/2))))
    if (np.abs(w_h[i]) > constraint):
      w_h[i] = np.sign(w_h[i]) * constraint

  hallucinated_result = np.matmul(E_h, w_h) + m_h
  hallucinated_result = hallucinated_result.reshape(width, height)
  return hallucinated_result

def get_weights(eigenfaces, input_image, mean_face):
  return np.matmul(eigenfaces.T, input_image - mean_face)

def main():
  LOW_RES_DATA_TRAINING_PATH = 'training/training_low_res.npy'
  HIGH_RES_DATA_TRAINING_PATH = 'training/training_high_res.npy'
  LOW_RES_DATA_TESTING_PATH = 'testing/testing_low_res.npy'
  HIGH_RES_DATA_TESTING_PATH = 'testing/testing_high_res.npy'

  ALPHA = .1 # Requires tuning

  low_res_images, high_res_images = load_data(LOW_RES_DATA_TRAINING_PATH, HIGH_RES_DATA_TRAINING_PATH)
  high_res_sample = scipy.misc.imread("../aligned_data/umass_aligned_images/_._data_umass_originalPics_2002_07_19_bigimg_18_aligned.png", mode='L')
  high_res_width, high_res_height = high_res_sample.shape

  # Change the input image, make sure it's a 1 x num_pixels image
  input_image = scipy.misc.imread("result/arvind_low_res.png", mode='L')
  input_image = input_image.reshape(32*24)
  plt.imshow(input_image.reshape(32, 24), cmap='gray')
  plt.show()

  low_res_weights = get_low_res_weights(
    images=low_res_images,
    input_image=input_image
  )

  hallucinated_result = get_hallucinated_results(
    images=high_res_images,
    width=high_res_width,
    height=high_res_height,
    low_res_weights=low_res_weights,
    alpha=ALPHA
  )

  plt.imshow(hallucinated_result, cmap="gray")
  plt.show()

if __name__ == "__main__":
  main()






import numpy as np 
from PIL import Image
from copy import deepcopy
from scipy import ndimage
import matplotlib.pyplot as plt

def load_data(low_res_data_path, high_res_data_path):
  low_res_images = np.load(low_res_file_path)
  high_res_images = np.load(high_res_file_path)

  return low_res_images, high_res_images

# Takes an 2D matrix consisted of N 1D image vectors and calculates the mean of all of the image vectors
def mean_face(face_matrix):
  return np.mean(face_matrix, axis=0)

def eigenfaces(face_matrix):
  return face_matrix - mean_face(face_matrix)

def orthonormal_eigenvectors(eigenfaces, eigenvectors, eigenvalues)
  pass




def main():
  LOW_RES_DATA_PATH = 'training/training_low_res.npy'
  HIGH_RES_DATA_PATH = 'training/training_high_res.npy'

  low_res_images, high_res_images = load_data(LOW_RES_DATA_PATH, HIGH_RES_DATA_PATH)
  
  m_l = np.mean(high_res_images, axis=0)
  L = high_res_images - m_l
  LL_T = np.matmul(L, L.T)
  # TO FINISH
main()






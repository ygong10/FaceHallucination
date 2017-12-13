import numpy as np
from PIL import Image
from copy import deepcopy
import scipy.linalg
import matplotlib.pyplot as plt

def load_data(low_res_file_path, high_res_file_path):
  low_res_images = np.load(low_res_file_path)
  high_res_images = np.load(high_res_file_path)

  return low_res_images, high_res_images

# Takes an 2D matrix consisted of N 1D image vectors and calculates the mean of all of the image vectors
def mean_face(face_matrix):
  return np.mean(face_matrix, axis=0)

def eigenfaces(face_matrix):
  return face_matrix - mean_face(face_matrix)

def orthonormal_eigenvectors(eigenfaces, eigenvectors, eigenvalues):
  pass




def main():
  LOW_RES_DATA_PATH = 'training/training_low_res.npy'
  HIGH_RES_DATA_PATH = 'training/training_high_res.npy'

  low_res_images, high_res_images = load_data(LOW_RES_DATA_PATH, HIGH_RES_DATA_PATH)

  input_image = low_res_images[:, 0]
  # test = input_image.reshape(16,12)
  # plt.imshow(test, cmap="gray")
  # plt.show()

  # PCA
  m_l = np.mean(low_res_images, axis=1)
  L = low_res_images - np.vstack(m_l)
  R = np.matmul(L.T, L)
  A_I, V_I = scipy.linalg.eig(R, left=True, right=False)
  # Turn the eigenvalues into an eigenvalue matrix
  A_I = np.eye(A_I.shape[0]) * A_I
  # Numpy returns complex value for a few pixels.
  E = np.real(np.matmul(np.matmul(L, V_I), (np.linalg.inv(A_I)**(1/2))))
  #result = E[:, 0]
  #result = result.reshape(29, 38)
  w_l = np.matmul(E.T, input_image - m_l)
  #result = np.matmul(E, w_l) + m_l
  #print(result.shape)
  # acc = 0
  # for i in range((E.shape[1])):
  #   if any(np.iscomplex(E[:,i])):
  #     acc+= 1
  # print(acc)
  # Numpy returns 0 complex for some reason.
  #result = np.real(E[:, 0])

  #print(result[0,0])
  #print(w_l.shape)
  #result = np.matmul(E, w_l) + m_l
  #result = result.reshape(454, 604)
  # plt.imshow(result, cmap="gray")
  # plt.show()
  # Eigentransformation
  m_h = np.mean(high_res_images, axis=1)
  x_h = np.zeros((128 * 96,))
  for i in range(len(w_l)):
    c_i = w_l[i]
    x_h += (c_i * high_res_images[:, i]) + m_h
  L_h = high_res_images - np.vstack(m_h)
  R_h = np.matmul(L_h.T, L_h)
  A_I_h, V_I_h = scipy.linalg.eig(R_h, left=True, right=False)
  A_I_h = np.eye(A_I_h.shape[0]) * A_I_h
  E_h = np.real(np.matmul(np.matmul(L_h, V_I_h), (np.linalg.inv(A_I_h)**(1/2))))
  #print(E_h.shape)
  w_h = np.matmul(E_h.T, x_h - m_h)
  print(w_h.shape)
  alpha = 0.5

  for i in range(len(w_h)):
    w_h_i = w_h[i]
    eigval = A_I_h[i,i]
    if (w_h_i > (alpha * (eigval**(1/2)))):
      w_h[i] = np.sign(w_h[i]) * (alpha * (eigval**(1/2)))

  hallucinated_result = np.matmul(E_h, w_h) + m_h
  hallucinated_result = hallucinated_result.reshape(128, 96)
  plt.imshow(hallucinated_result, cmap="gray")
  plt.show()


main()






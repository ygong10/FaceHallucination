import numpy as np 
from PIL import Image
from copy import deepcopy

# Converts a pgm image into a numpy ndarray
def convertPgm(pgmImg):
  im = Image.open(pgmImg)
  width, height = im.size
  return np.asarray(im).reshape(width, height)

# Subsample a numpy ndarray image by an integer factor
def subsample(im, factor):
  width, height = im.shape
  if (factor > width or factor > height):
    raise ValueError("Subsample factor is bigger than image dimensions")
    
  im = deepcopy(im)
  im = im[::factor, ::factor]
  return im

im = convertPgm("../data/CAFE-FACS-Orig/004_d2.pgm")
im =subsample(im)

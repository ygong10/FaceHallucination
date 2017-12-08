import numpy as np 
from PIL import Image

# Converts a pgm image into a numyp ndarray
def convertPgm(pgmImg):
  im = Image.open(pgmImg)
  width, height = im.size
  return np.asarray(im).reshape(width, height)

im = convertPgm("data/CAFE-FACS-Orig/004_d2.pgm")

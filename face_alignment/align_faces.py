# USAGE
# python align_faces.py -f <path_to_face_images_directory> -a <path_to_aligned_faces_directory>

# EXAMPLE
# python align_faces.py -f ../data/CAFE-FACS-Orig -a ../aligned_data/ucsd_aligned_images

# import the necessary packages
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import argparse
import imutils
import dlib
import cv2
import os

def align_faces(shape, path_to_images, image_name, path_to_aligned_images, rows, cols):
	# initialize dlib's face detector (HOG-based) and then create
	# the facial landmark predictor and the face aligner
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(shape)
	fa = FaceAligner(predictor, desiredFaceWidth=cols, desiredFaceHeight=rows)

	# load the input image, resize it, and convert it to grayscale
	image = cv2.imread(os.path.join(path_to_images, image_name))
	image = imutils.resize(image, width=800)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	rects = detector(gray, 2)
	if(len(rects) > 0):
		rect = rects[0]
		(x, y, w, h) = rect_to_bb(rect)
		faceOrig = imutils.resize(image[y:y + h, x:x + w], width=cols, height=rows)
		faceAligned = fa.align(image, gray, rect)

		if path_to_aligned_images[-1] != '/':
			path_to_aligned_images += '/'
		cv2.imwrite(path_to_aligned_images + image_name + "_aligned.png", faceAligned)

if __name__=="__main__":
	rows = 128
	cols = 96
	ap = argparse.ArgumentParser()
	ap.add_argument("-f", "--faces", required=True, help="path to directory of face images")
	ap.add_argument("-a", "--aligned_faces", required=True, help="path to directory to save aligned images")
	args = vars(ap.parse_args())
	path_to_images = args["faces"]
	path_to_aligned_images = args["aligned_faces"]
	path_to_shape = 'shape_predictor_68_face_landmarks.dat'
	for filename in os.listdir(path_to_images):
		align_faces(path_to_shape, path_to_images, filename, path_to_aligned_images, rows, cols)
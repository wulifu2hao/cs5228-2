import cv2
from os.path import isdir, join, isfile, splitext, basename
from os import makedirs, listdir
from random import sample
import numpy as np
from scipy.io import loadmat, savemat
from datetime import datetime


def get_classes(datasetpath):
	classes = [files for files in listdir(datasetpath) if isdir(join(datasetpath, files))]
	classes.sort()
	if len(classes) == 0:
		raise ValueError('no classes found')
	return classes

def standarizeImage(im):
    if im.shape[0] > 480:
        resize_factor = 480.0 / im.shape[0]
        im = cv2.resize(im, (0,0), fx = resize_factor, fy = resize_factor)
    return im

def detect_sift_features(imagepath, detector):
	im = cv2.imread(imagepath, 0)
	im = standarizeImage(im)
	kps, descrs = detector.detectAndCompute(im, None)
	return (kps, descrs, im.shape)

def read_standarized_image_full(imagepath):
	im = cv2.imread(imagepath,0)
	return standarizeImage(im)

def drawMatches(img1, kp1, img2, kp2, matches):

	rows1 = img1.shape[0]
	cols1 = img1.shape[1]
	rows2 = img2.shape[0]
	cols2 = img2.shape[1]

	out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

	# Place the first image to the left
	out[:rows1,:cols1] = np.dstack([img1, img1, img1])

	# Place the next image to the right of it
	out[:rows2,cols1:] = np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
	for mat in matches:

		# Get the matching keypoints for each of the images
		img1_idx = mat.queryIdx
		img2_idx = mat.trainIdx

		# x - columns
		# y - rows
		(x1,y1) = kp1[img1_idx].pt
		(x2,y2) = kp2[img2_idx].pt

		# Draw a small circle at both co-ordinates
		# radius 4
		# colour blue
		# thickness = 1
		cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)   
		cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

		# Draw a line in between the two points
		# thickness = 1
		# colour blue
		cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)


	# Show the image
	# cv2.imshow('Matched Features', out)
	# cv2.waitKey(0)
	# cv2.destroyWindow('Matched Features')

	return out

working_dir = "test_dir"
train_dir = join(working_dir, "train")
query_dir = join(working_dir, "query")
result_dir = join(working_dir, "result")
if not isdir(result_dir):
	makedirs(result_dir)

acceptable_image_formats = [".jpg", ".png", ".jpeg"]
detector = cv2.SIFT()
matcher = cv2.BFMatcher()



classes = get_classes(train_dir)

for class_idx, m_class in enumerate(classes):
	# if class_idx != 0:
	# 	continue
	class_path = join(train_dir, m_class)
	onlyfiles = [join(class_path,f) for f in listdir(class_path) if isfile(join(class_path,f)) and splitext(f)[1].lower() in acceptable_image_formats]
	for x in xrange(0,30):
		
		pair = sample(onlyfiles, 2)	
		query_image_path = pair[0].strip()
		train_image_path = pair[1].strip()
		(query_kps, query_descrs, _) = detect_sift_features(query_image_path, detector)
		(train_kps, train_descrs, _) = detect_sift_features(train_image_path, detector)
		matches = matcher.knnMatch(query_descrs, train_descrs, k=2)

		if len(query_kps) == 0 or len(train_kps) == 0:
			continue 

		good = []
		for m,n in matches:
		    if m.distance < 0.7*n.distance:
		        good.append(m)

		good = [m for (m,n) in matches]
		good.sort(key=lambda m: m.distance)
		good = good[:10]

		query_im = read_standarized_image_full(query_image_path)
		train_im = read_standarized_image_full(train_image_path)
		result = drawMatches(query_im,query_kps,train_im,train_kps,good)
		output_dir = join(result_dir, "pair_match", "%d"%class_idx)
		if not isdir(output_dir):
			makedirs(output_dir)
		cv2.imwrite(join(output_dir, "%d_%d.jpg"%(x, len(good))), result)


		





























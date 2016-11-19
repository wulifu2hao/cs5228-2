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
	im = cv2.imread(imagepath)
	return standarizeImage(im)

working_dir = "test_dir"
train_dir = join(working_dir, "train")
query_dir = join(working_dir, "query")
result_dir = join(working_dir, "result")
if not isdir(result_dir):
	makedirs(result_dir)

acceptable_image_formats = [".jpg", ".png", ".jpeg"]
detector = cv2.SIFT()


classes = get_classes(train_dir)

train_image_path = []
for class_idx, m_class in enumerate(classes):
	class_path = join(train_dir, m_class)
	onlyfiles = [join(class_path,f) for f in listdir(class_path) if isfile(join(class_path,f)) and splitext(f)[1].lower() in acceptable_image_formats]
	train_image_path.extend(onlyfiles)

croped_sample_sift_demo_dir = join(result_dir, "cropped_sift_demo")
if not isdir(croped_sample_sift_demo_dir):
	makedirs(croped_sample_sift_demo_dir)

SAMPLE_SIZE = 100
train_image_path = sample(train_image_path, SAMPLE_SIZE)
average_sift_count = 0
for image_path in train_image_path:
	(kps, _, _) =  detect_sift_features(image_path, detector)
	average_sift_count += float(len(kps))/SAMPLE_SIZE
	query_im = read_standarized_image_full(image_path)
	for kp in kps:
		px, py = kp.pt
		cv2.circle(query_im, (int(px), int(py)), 5, (0,255,0))
	cv2.imwrite(join(croped_sample_sift_demo_dir, basename(image_path)),query_im)

print average_sift_count





















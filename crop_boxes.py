import cv2
# import sys
from os.path import isdir, join, isfile, splitext, basename
from os import makedirs, listdir
# from random import sample
import numpy as np
# from sklearn.cluster.k_means_ import k_means
from scipy.io import loadmat, savemat
# from scipy.cluster.vq import vq
# from itertools import groupby as g
from datetime import datetime
import xml.etree.ElementTree

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
	# print imagepath
	im = cv2.imread(imagepath)
	# print im
	return standarizeImage(im)

def draw_grid(im, unit, row, col):
	height_unit = im.shape[0]/unit + 1
	width_unit = im.shape[1]/unit + 1
	x_min = col*width_unit + 3
	x_max = (col+1)*width_unit - 3
	y_min = row*height_unit + 3
	y_max = (row+1)*height_unit - 3
	pts = np.int32([[x_min, y_min], [x_max,y_min], [x_max,y_max], [x_min, y_max]]).reshape(-1,1,2)
	# cv2.polylines(im, pts, True, (255,255,255), 10, cv2.CV_AA)

	pt1 = (x_min, y_min)
	pt2 = (x_max,y_min)
	pt3 = (x_max,y_max)
	pt4 = (x_min,y_max)
	cv2.line(im, pt1, pt2, (255,255,255))
	cv2.line(im, pt2, pt3, (255,255,255))
	cv2.line(im, pt3, pt4, (255,255,255))
	cv2.line(im, pt4, pt1, (255,255,255))


	return im

def draw_box(im, x_min, y_min, x_max, y_max):
	pt1 = (x_min, y_min)
	pt2 = (x_max,y_min)
	pt3 = (x_max,y_max)
	pt4 = (x_min,y_max)
	cv2.line(im, pt1, pt2, (255,255,255))
	cv2.line(im, pt2, pt3, (255,255,255))
	cv2.line(im, pt3, pt4, (255,255,255))
	cv2.line(im, pt4, pt1, (255,255,255))

	return im



working_dir = "test_dir"
train_dir = join(working_dir, "train")
query_dir = join(working_dir, "query")
result_dir = join(working_dir, "result")
if not isdir(result_dir):
	makedirs(result_dir)

acceptable_image_formats = [".jpg", ".png", ".jpeg"]
CHOP_UNIT = 10
NUM_OF_PIECES = CHOP_UNIT*CHOP_UNIT

detector = cv2.SIFT()
# detector = cv2.SIFT(nfeatures=1000)


classes = get_classes(train_dir)

train_image_path = []
train_image_classes = []
class_mapping = []
average_num_images_per_class = 0
for class_idx, m_class in enumerate(classes):
	class_mapping.append(m_class)
	class_path = join(train_dir, m_class)
	onlyfiles = [ join(class_path,f) for f in listdir(class_path) if isfile(join(class_path,f)) and splitext(f)[1].lower() in acceptable_image_formats]
	labels = [class_idx for _ in onlyfiles]
	train_image_path.extend(onlyfiles)
	train_image_classes.extend(labels)
	average_num_images_per_class += (float(len(onlyfiles))/len(classes))
average_num_images_per_class = int(average_num_images_per_class)

if isfile(join(result_dir, "sift_features.mat")):
	obj = loadmat(join(result_dir, "sift_features.mat"))
	sift_inverted_indexes = obj['sift_inverted_indexes']
	sift_features = obj['sift_features']
	# train_image_path = obj['train_image_path']
	# train_image_classes = obj['train_image_classes'][0]
	# class_mapping = obj['class_mapping']
	# print sift_features.shape
	# print sift_inverted_indexes.shape
	# print train_image_path.shape
	# print train_image_classes.shape
	# print class_mapping.shape
else:
	sift_inverted_indexes = []
	sift_features = []
	
	for train_image_idx, image_path in enumerate(train_image_path):
		kps, descrs, shape = detect_sift_features(image_path, detector)
		height_unit = shape[0]/CHOP_UNIT + 1
		width_unit = shape[1]/CHOP_UNIT + 1
		for kp_idx, kp in enumerate(kps):
			px, py = kp.pt
			row = int(py)/height_unit
			col = int(px)/width_unit
			piece_idx = row*CHOP_UNIT + col
			sift_inverted_indexes.append((train_image_idx, piece_idx))
		sift_features.append(descrs)
	sift_features = np.concatenate(sift_features, axis=0)
	print sift_features.shape
	savemat(join(result_dir, "sift_features.mat"), {
		"sift_features":sift_features,
		"sift_inverted_indexes":sift_inverted_indexes,
		"train_image_path":train_image_path,
		"train_image_classes":train_image_classes,
		"class_mapping":class_mapping,
		"average_num_images_per_class":average_num_images_per_class
		})

show_box_dir = join(result_dir, "boxes")

for image_idx, test_image_path in enumerate(train_image_path):
	test_image_path = test_image_path.strip()
	class_folder_name = class_mapping[train_image_classes[image_idx]]
	show_box_class_dir = join(show_box_dir, class_folder_name)
	if not isdir(show_box_class_dir):
		makedirs(show_box_class_dir)

	box_dir = join(working_dir, "boxes", class_folder_name)
	box_file_name = join(box_dir, splitext(basename(test_image_path))[0]+".xml")
	query_im = cv2.imread(test_image_path)
	box = None
	num_of_boxes = 0
	if isfile(box_file_name):
		e = xml.etree.ElementTree.parse(box_file_name).getroot()
		objs = []
		for child in e:
			if child.tag == "object":
				num_of_boxes += 1
				box = child

	if num_of_boxes == 1:
		bndbox = box[4]
		xmin = int(bndbox[0].text)
		ymin = int(bndbox[1].text)
		xmax = int(bndbox[2].text)
		ymax = int(bndbox[3].text)
		query_im = query_im[ymin:ymax, xmin:xmax]
		cv2.imwrite(join(show_box_class_dir, basename(test_image_path)),query_im)



# let's take an image, divide by 100 parts
# for each piece, take its sift features, just find 1nn and cast vote
# the piece with highest vote will win



# for each image, divide the sift features into 100 pieces
# for each piece, finds its knn, if the majority is from the same class then it's good 
	# value of k?
	# what is majority?












	
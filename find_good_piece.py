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
	x_min = col*width_unit
	x_max = (col+1)*width_unit
	y_min = row*height_unit
	y_max = (row+1)*height_unit
	pts = np.int32([[x_min, y_min], [x_max,y_min], [x_max,y_max], [x_min, y_max]]).reshape(-1,1,2)
	cv2.polylines(im, pts, True, (255,255,255), 10, cv2.CV_AA)
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

detector = cv2.SIFT(nfeatures=1000)



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
	train_image_path = obj['train_image_path']
	print sift_features.shape
	print sift_inverted_indexes.shape
	print train_image_path.shape
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

test_image_path = join(train_dir, "n01440764", "n01440764_39.JPEG")
kps, descrs, shape = detect_sift_features(test_image_path, detector)
height_unit = shape[0]/CHOP_UNIT + 1
width_unit = shape[1]/CHOP_UNIT + 1

# for test only
# test_im = standarizeImage(cv2.imread(test_image_path))


test_image_features_in_pieces = [[] for x in range(CHOP_UNIT*CHOP_UNIT)]


for kp_idx, kp in enumerate(kps):
	px, py = kp.pt

	# for test only
	# cv2.circle(test_im, (int(px), int(py)), 5, (0,255,0))

	row = int(py)/height_unit
	col = int(px)/width_unit
	piece_idx = row*CHOP_UNIT + col
	test_image_features_in_pieces[piece_idx].append(descrs[kp_idx])

# for test only
# dist = [len(features) for features in test_image_features_in_pieces]
# print dist
# cv2.imwrite(join(result_dir, "test_kp_position.jpg"), test_im)

test_image_result_dir = join(result_dir, "piece_knn")
if not isdir(test_image_result_dir):
	makedirs(test_image_result_dir)

matcher = cv2.BFMatcher()
for piece_idx, sift_feature_of_cur_piece in enumerate(test_image_features_in_pieces):

	sift_feature_of_cur_piece = np.array(sift_feature_of_cur_piece, dtype=np.float32)
	matches = matcher.knnMatch(sift_feature_of_cur_piece, sift_features, k=2)
	# matches = matcher.match(sift_feature_of_cur_piece, sift_features)
	vote_matrix = np.zeros((len(train_image_path),NUM_OF_PIECES))
	for match in matches:
		match = match[1]
		(train_image_idx, train_piece_idx) = sift_inverted_indexes[match.trainIdx]
		vote_matrix[train_image_idx][train_piece_idx] += 1
	best_idx = np.argmax(vote_matrix)
	best_image_idx = best_idx/NUM_OF_PIECES
	best_piece_idx = best_idx%NUM_OF_PIECES



	cur_dir = join(test_image_result_dir, "%d"%piece_idx)
	if not isdir(cur_dir):
		makedirs(cur_dir)

	query_piece_row = piece_idx/CHOP_UNIT
	query_piece_col = piece_idx%CHOP_UNIT
	query_im = read_standarized_image_full(test_image_path)
	query_im = draw_grid(query_im, CHOP_UNIT, query_piece_row, query_piece_col)
	cv2.imwrite(join(cur_dir, "query.jpg"),query_im)


	best_piece_row = best_piece_idx/CHOP_UNIT
	best_piece_col = best_piece_idx%CHOP_UNIT
	# if train_image_path[best_image_idx].strip() == "test_dir/train/n01484850/n01484850_76.JPEG":
		# print "skip"
		# continue
	result_im = read_standarized_image_full(train_image_path[best_image_idx].strip())
	result_im = draw_grid(result_im, CHOP_UNIT, best_piece_row, best_piece_col)
	cv2.imwrite(join(cur_dir, "result.jpg"),result_im)
	




# let's take an image, divide by 100 parts
# for each piece, take its sift features, just find 1nn and cast vote
# the piece with highest vote will win



# for each image, divide the sift features into 100 pieces
# for each piece, finds its knn, if the majority is from the same class then it's good 
	# value of k?
	# what is majority?












	
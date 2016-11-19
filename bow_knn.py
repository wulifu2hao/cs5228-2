import cv2
import sys
from os.path import isdir, join, isfile, splitext, basename
from os import makedirs, listdir
from random import sample
import numpy as np
from sklearn.cluster.k_means_ import k_means
from scipy.io import loadmat, savemat
from scipy.cluster.vq import vq
from itertools import groupby as g
from datetime import datetime
import os
from shutil import copy2

def most_common_oneliner(L):
  return max(g(sorted(L)), key=lambda(x, v):(len(list(v)),-L.index(x)))[0]

def sort_by_frequency(L):
	return [x for (x,y) in sorted(g(sorted(L)), key=lambda(x, v):(len(list(v)),-L.index(x)))]	

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
	return descrs


if __name__ == "__main__":
	if len(sys.argv) >= 3:
		working_dir = sys.argv[1]
		
		try:
			option = int(sys.argv[2])
		except Exception, e:
			raise ValueError("error: please specific the options:\n1. direct knn\n2. direct svm\n3. lamp svm\n4. direct CNN\n5. lamp CNN")
		else:
			pass
		finally:
			pass

		if not isdir(working_dir):
			raise ValueError("error: directory not exist")
		elif not isdir(join(working_dir, "train")):
			raise ValueError("error: training dataset directory not found")
		elif not isdir(join(working_dir, "query")):
			raise ValueError("error: query dataset directory not found")
		else:
			acceptable_image_formats = [".jpg", ".png", ".jpeg"]
			FRACTION_OF_IMAGES_FOR_VOCAB = 10
			MAX_IAMGES_NUM_FOR_VOCAB = 300
			detector = cv2.SIFT(nfeatures=1000)
			NUM_OF_WORD_FOR_VOCAB = 1000
			MAX_NUM_OF_FEATURES_FOR_VOCAB = 20000

			train_dir = join(working_dir, "train")
			query_dir = join(working_dir, "query")
			result_dir = join(working_dir, "result")
			if not isdir(result_dir):
				makedirs(result_dir)

			classes = get_classes(train_dir)

			# train vocabulary, take 1/10 of the images, at most 100 images 
			if not isfile(join(result_dir, "vocab.mat")):
				print "training vocabulary. This may take a while, but the result will be saved for future usage"
				images_path_for_vocab = []
				for m_class in classes:
					class_path = join(train_dir, m_class)
					onlyfiles = [join(class_path,f) for f in listdir(class_path) if isfile(join(class_path,f)) and splitext(f)[1].lower() in acceptable_image_formats]
					images_path_for_vocab.extend(onlyfiles)

				images_path_for_vocab = sample(images_path_for_vocab, len(images_path_for_vocab)/FRACTION_OF_IMAGES_FOR_VOCAB+1)
				if len(images_path_for_vocab) > MAX_IAMGES_NUM_FOR_VOCAB:
					images_path_for_vocab = sample(images_path_for_vocab, MAX_IAMGES_NUM_FOR_VOCAB)

				descrs_for_vocab = []
				for image_path in images_path_for_vocab:
					descrs = detect_sift_features(image_path, detector)
					if descrs is not None:
						descrs_for_vocab.append(descrs)
				descrs_for_vocab = list(np.concatenate(descrs_for_vocab, axis=0))
				if len(descrs_for_vocab) > MAX_NUM_OF_FEATURES_FOR_VOCAB:
					descrs_for_vocab = sample(descrs_for_vocab, MAX_NUM_OF_FEATURES_FOR_VOCAB)
				descrs_for_vocab = np.array(descrs_for_vocab, dtype=np.float64)
				# print descrs_for_vocab.shape

				# result = []
				# for x in xrange(0,30):
				# 	print x
				# 	num_of_cluster = 1+20*x
				# 	_, _, inertia_  = k_means(descrs_for_vocab, num_of_cluster)
				# 	result.append([num_of_cluster, inertia_])

				# import matplotlib.pyplot as plt
				# plt.plot(*zip(*result))
				# plt.show()

				print "clustering sift features to form vocabulary"
				print datetime.now()
				vocab, _, _ = k_means(descrs_for_vocab, NUM_OF_WORD_FOR_VOCAB, verbose=True)
				savemat(join(result_dir, "vocab.mat"),{"vocab":vocab})
			
			else:
				vocab = loadmat(join(result_dir, "vocab.mat"))['vocab']
				print vocab.shape



			# extract sift features, dowmsample if needed, convert to BOW
			# 1000 sift features * 128dim * 2byte -> 4 images per MB ->  4000 images per GB
			if not isfile(join(result_dir, "train_bow.mat")):
				print "computing bag of word representation for train images. This may take a while, but the result will be saved for future usage"
				train_image_path = []
				train_image_classes = []
				query_image_path = []
				query_image_classes = []
				class_mapping = []
				average_num_images_per_class = 0

				for class_idx, m_class in enumerate(classes):
					class_mapping.append(m_class)
					class_path = join(train_dir, m_class)
					onlyfiles = [ join(class_path,f) for f in listdir(class_path) if isfile(join(class_path,f)) and splitext(f)[1].lower() in acceptable_image_formats]
					labels = [class_idx for _ in onlyfiles]

					query_sample_idx = sample([x for x in xrange(0,len(onlyfiles))], len(onlyfiles)/10)
					train_sample_idx = [x for x in xrange(0,len(onlyfiles)) if x not in query_sample_idx]
					query_files = [onlyfiles[idx] for idx in query_sample_idx]
					query_labels = [labels[idx] for idx in query_sample_idx]
					train_files = [onlyfiles[idx] for idx in train_sample_idx]
					train_labels = [labels[idx] for idx in train_sample_idx]

					query_image_path.extend(query_files)
					query_image_classes.extend(query_labels)
					train_image_path.extend(train_files)
					train_image_classes.extend(train_labels)
					average_num_images_per_class += (float(len(onlyfiles))/len(classes))
				average_num_images_per_class = int(average_num_images_per_class)

				ONE_PERCENT = len(train_image_path)/100
				if ONE_PERCENT == 0:
					ONE_PERCENT = 1
				train_bows = []
				for train_image_idx, image_path in enumerate(train_image_path):
					if train_image_idx % ONE_PERCENT == 0:
						print "processing train images: %d percent" % (train_image_idx/ONE_PERCENT)
					descrs = detect_sift_features(image_path, detector)
					if descrs is not None:
						counts, _ = vq(descrs, vocab)
					else:
						counts = []
					bow,_ = np.histogram(counts, bins=[x for x in xrange(0,NUM_OF_WORD_FOR_VOCAB+1)])
					train_bows.append(bow)
					# print bow
				train_bows = np.array(train_bows, dtype=np.float32)

				savemat(join(result_dir, "train_bow.mat"),
					{"train_image_path":train_image_path,
					"train_image_classes":train_image_classes,
					"class_mapping":class_mapping,
					"train_bows":train_bows, 
					"average_num_images_per_class":average_num_images_per_class})

				ONE_PERCENT = len(query_image_path)/100
				if ONE_PERCENT == 0:
					ONE_PERCENT = 1
				query_bows = []
				for query_image_idx, image_path in enumerate(query_image_path):
					if query_image_idx % ONE_PERCENT == 0:
						print "processing query images: %d percent" % (query_image_idx/ONE_PERCENT)
					descrs = detect_sift_features(image_path, detector)
					if descrs is not None:
						counts, _ = vq(descrs, vocab)
					else:
						counts = []
					bow,_ = np.histogram(counts, bins=[x for x in xrange(0,NUM_OF_WORD_FOR_VOCAB+1)])
					query_bows.append(bow)
				query_bows = np.array(query_bows, dtype=np.float32)
				savemat(join(result_dir, "query_bow.mat"),
					{"query_image_path":query_image_path, 
					"query_image_classes":query_image_classes,
					"query_bows":query_bows})
			else:
				obj = loadmat(join(result_dir, "train_bow.mat"))
				train_image_path = obj['train_image_path']
				train_image_classes = obj['train_image_classes'][0]
				class_mapping = obj['class_mapping']
				train_bows = obj['train_bows']
				average_num_images_per_class = obj['average_num_images_per_class']

				obj = loadmat(join(result_dir, "query_bow.mat"))
				query_image_path = obj['query_image_path']
				query_bows = obj['query_bows']
				query_image_classes = obj['query_image_classes'][0]

			# classify query images

			# direct knn
			if option == 1:
				matcher = cv2.BFMatcher()
				matches = matcher.knnMatch(query_bows, train_bows, k=average_num_images_per_class)
				result_file = open(join(result_dir, "direct_knn.txt"), "w")
				accuracy_file = open(join(result_dir, "accuracy_bowknn.txt"), "w") # for testing only
				average_percentage = 0 # for testing only
				extreme_dir = join(result_dir, "knn_extreme")
				for query_idx, knnMatch in enumerate(matches):
					neighbour_classes = [train_image_classes[match.trainIdx] for match in knnMatch]
					predicted_classes = sort_by_frequency(neighbour_classes)
					if len(predicted_classes) > 5:
						predicted_classes = predicted_classes[:5]
					result_file.write("%s: "%basename(query_image_path[query_idx]))
					for predicted_class in predicted_classes:
						result_file.write("%s "%class_mapping[predicted_class])
					result_file.write("\n")


					# This block is for testing only 
					class_mapping = list(class_mapping)
					# ground_truth = basename(query_image_path[query_idx]).split("_")[0]
					# ground_truth = class_mapping.index(ground_truth)
					ground_truth = query_image_classes[query_idx]
					percentage = float(neighbour_classes.count(ground_truth))/len(neighbour_classes)
					if percentage < 0.05:
						copy2(query_image_path[query_idx].strip(), join(extreme_dir, "bad", basename(query_image_path[query_idx].strip())))
					if percentage > 0.25:
						copy2(query_image_path[query_idx].strip(), join(extreme_dir, "good", basename(query_image_path[query_idx].strip())))
						
					average_percentage += (percentage/len(query_image_path))
					prediction = most_common_oneliner(neighbour_classes)
					is_prediction_correct = "wrong"
					if ground_truth == prediction:
						is_prediction_correct = "correct"
					accuracy_file.write("%s: percentage is %f, predicted as %s, is %s\n" %(basename(query_image_path[query_idx]), percentage, class_mapping[prediction], is_prediction_correct))



				result_file.close()
				accuracy_file.write("average percentage: %f"%average_percentage) # for testing only
				accuracy_file.close() # for testing only
			
			# direct svm
			if option == 2:
				print query_bows.shape
				print train_bows.shape 
				print train_image_classes.shape
				print class_mapping.shape

				result_file = open(join(result_dir, "direct_svm.txt"), "w")

				result_file.close()

			# lamp svm
			if option == 3:
				matcher = cv2.BFMatcher()
				matches = matcher.knnMatch(query_bows, train_bows, k=average_num_images_per_class)

				result_file = open(join(result_dir, "lamp_svm.txt"), "w")
				for query_idx, knnMatch in enumerate(matches):
					neighbour_classes = [train_image_classes[match.trainIdx] for match in knnMatch]
					neighbour_bow = [train_bows[match.trainIdx] for match in knnMatch]

					# train svm and predict

					# write the result into file

				result_file.close()

			# direct CNN
			if option == 4:
				print query_image_path.shape
				print train_image_path.shape 
				print train_image_classes.shape
				print class_mapping.shape

				# generate train.txt
				cnn_dir = "CNN"
				if not isdir(cnn_dir):
					makedirs(cnn_dir)
				train_file = open(join(cnn_dir, "train.txt"), "w")
				for i in range(len(train_image_path)):
					train_file.write(str(train_image_path[i]) + " " + str(train_image_classes[i]) + "\n")
				train_file.close()
				val_file = open(join(cnn_dir, "val.txt"), "w")
				for i in range(len(train_image_path)):
					val_file.write(str(train_image_path[i]) + " " + str(train_image_classes[i]) + "\n")
				val_file.close()

				os.chdir(cnn_dir)
				result_dir = join("..", result_dir)
				with open("run_train_imagenet.bat", "wb") as f:
					f.write("@echo off\r\ncall train_imagenet.bat >" + join(result_dir, "direct_CNN.txt") + ".txt 2>&1")
					f.close()
				with open("autorun.bat", "rb") as f:
					for command in f:
						command = command.strip()
				 		os.system(command)
				 	f.close()
				
				# result_file = open(join(result_dir, "direct_CNN.txt"), "w")

				# result_file.close()

			# lamp CNN
			if option == 5:
				matcher = cv2.BFMatcher()
				matches = matcher.knnMatch(query_bows, train_bows, k=average_num_images_per_class)
				count = 0
				#result_file = open(join(result_dir, "lamp_CNN.txt"), "w")
				for query_idx, knnMatch in enumerate(matches):
					count += 1
					neighbour_classes = [train_image_classes[match.trainIdx] for match in knnMatch]
					neighbour_path = [train_image_path[match.trainIdx] for match in knnMatch]

					cnn_dir = "CNN"
					if not isdir(cnn_dir):
						makedirs(cnn_dir)
					train_file = open(join(cnn_dir, "train.txt"), "w")
					for i in range(len(neighbour_path)):
						train_file.write(str(neighbour_path[i]) + " " + str(neighbour_classes[i]) + "\n")
					train_file.close()
					val_file = open(join(cnn_dir, "val.txt"), "w")
					for i in range(len(neighbour_path)):
						val_file.write(str(neighbour_path[i]) + " " + str(neighbour_classes[i]) + "\n")
					val_file.close()

					os.chdir(cnn_dir)
					result_dir = join("..", result_dir)
					with open("run_train_imagenet.bat", "wb") as f:
						f.write("@echo off\r\ncall train_imagenet.bat >" + join(result_dir, "lamp_CNN" + str(count)) + ".txt 2>&1")
						f.close()
					with open("autorun.bat", "rb") as f:
						for command in f:
							command = command.strip()
					 		os.system(command)
					 	f.close()
					os.chdir("../")
				#result_file.close()

	elif len(sys.argv) == 1: 
		raise ValueError("error: please specific the working directory")
	else:
		raise ValueError("error: please specific the options:\n1. direct knn\n2. direct svm\n3. lamp svm\n4. direct CNN\n5. lamp CNN")

	


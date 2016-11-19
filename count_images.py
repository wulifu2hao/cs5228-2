from os.path import isdir, join, isfile, splitext, basename
from os import makedirs, listdir
import numpy as np

def get_classes(datasetpath):
	classes = [files for files in listdir(datasetpath) if isdir(join(datasetpath, files))]
	classes.sort()
	if len(classes) == 0:
		raise ValueError('no classes found')
	return classes



working_dir = "test_dir"
train_dir = join(working_dir, "train")
query_dir = join(working_dir, "query")
result_dir = join(working_dir, "result")
if not isdir(result_dir):
	makedirs(result_dir)

acceptable_image_formats = [".jpg", ".png", ".jpeg"]

classes = get_classes(train_dir)
total_count = 0
for class_idx, m_class in enumerate(classes):
	class_path = join(train_dir, m_class)
	onlyfiles = [join(class_path,f) for f in listdir(class_path) if isfile(join(class_path,f)) and splitext(f)[1].lower() in acceptable_image_formats]
	print "%s: %d"%(m_class, len(onlyfiles))
	total_count += len(onlyfiles)

print total_count
# detect OBSEA in photos with mask rcnn model
import os.path
from os import listdir
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from numpy import expand_dims
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.model import mold_image
from mrcnn.utils import Dataset
# import numpy as np
import json
from tqdm.auto import tqdm
from rich import print as rprint
import time


STEPS_PER_EPOCH = 3366


# class that defines and loads the OBSEA dataset
class OBSEADataset(Dataset):
	# load the dataset definitions
	def load_dataset(self, dataset_dir, is_train=True):
		# define one class
		self.add_class("dataset", 1, "OBSEA")
		# define data locations
		images_dir = dataset_dir + '/Tagging_Img/'
		annotations_dir = dataset_dir + '/Tagging_XML/'
		# find all images
		'''Modificació'''
		# print(sorted(listdir(images_dir), key=lambda s:int(s.split('.')[0])))

		for filename in sorted(listdir(images_dir), key=lambda s: int(s.split('.')[0])):
			print(filename)
			# extract image id
			image_id = filename[:-4]
			# skip bad images
			if image_id in ['090']:
				continue
			# skip all images after 150 if we are building the train set
			if is_train and int(image_id) >= STEPS_PER_EPOCH:
				continue
			# skip all images before 150 if we are building the test/val set
			if not is_train and int(image_id) < STEPS_PER_EPOCH:
				continue
			img_path = images_dir + filename
			ann_path = annotations_dir + image_id + '.xml'
			# add to dataset
			self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)

	# load all bounding boxes for an image
	def extract_boxes(self, filename):
		# load and parse the file
		root = ElementTree.parse(filename)
		boxes = list()
		# extract each bounding box
		for box in root.findall('.//bndbox'):
			xmin = int(box.find('xmin').text)
			ymin = int(box.find('ymin').text)
			xmax = int(box.find('xmax').text)
			ymax = int(box.find('ymax').text)
			coors = [xmin, ymin, xmax, ymax]
			boxes.append(coors)
		# extract image dimensions
		width = int(root.find('.//size/width').text)
		height = int(root.find('.//size/height').text)
		return boxes, width, height

	# load the masks for an image
	def load_mask(self, image_id):
		# get details of image
		info = self.image_info[image_id]
		# define box file location
		path = info['annotation']
		# load XML
		boxes, w, h = self.extract_boxes(path)
		# create one array for all masks, each on a different channel
		masks = zeros([h, w, len(boxes)], dtype='uint8')
		# create masks
		class_ids = list()
		for i in range(len(boxes)):
			box = boxes[i]
			row_s, row_e = box[1], box[3]
			col_s, col_e = box[0], box[2]
			masks[row_s:row_e, col_s:col_e, i] = 1
			class_ids.append(self.class_names.index('OBSEA'))
		return masks, asarray(class_ids, dtype='int32')

	# load an image reference
	def image_reference(self, image_id):
		info = self.image_info[image_id]
		return info['path']


# define the prediction configuration
class PredictionConfig(Config):
	# define the name of the configuration
	NAME = "OBSEA_cfg"
	# number of classes (background + OBSEA)
	NUM_CLASSES = 1 + 1
	# simplify GPU config
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1


# plot a number of photos with ground truth and predictions
def plot_actual_vs_predicted(dataset, model, cfg, n_images=5):
	# load image and mask
	for i in range(n_images):
		# load the image and mask
		image = dataset.load_image(i)
		mask, _ = dataset.load_mask(i)
		# convert pixel values (e.g. center)
		scaled_image = mold_image(image, cfg)
		# convert image into one sample
		sample = expand_dims(scaled_image, 0)
		# make prediction
		yhat = model.detect(sample, verbose=0)[0]
		# define subplot
		pyplot.subplot(n_images, 2, i*2+1)
		# plot raw pixel data
		pyplot.imshow(image)
		pyplot.title('Actual')
		# plot masks
		for j in range(mask.shape[2]):
			pyplot.imshow(mask[:, :, j], cmap='gray', alpha=0.3)
		# get the context for drawing boxes
		pyplot.subplot(n_images, 2, i*2+2)
		# plot raw pixel data
		pyplot.imshow(image)
		pyplot.title('Predicted')
		ax = pyplot.gca()
		# plot each box
		for box in yhat['rois']:
			# get coordinates
			y1, x1, y2, x2 = box
			# calculate width and height of the box
			width, height = x2 - x1, y2 - y1
			# create the shape
			rect = Rectangle((x1, y1), width, height, fill=False, color='red')
			# draw the box
			ax.add_patch(rect)
	# show the figure
	pyplot.show()


def plot_actual_vs_predicted_2(dataset, model, cfg, type_dataset, n_images=10):
	# load image and mask
	global rect
	for i in tqdm(range(n_images)):
		# load the image and mask
		t = time.time()
		image = dataset.load_image(i)
		time1 = time.time()-t
		t = time.time()
		mask = []
		mask, _ = dataset.load_mask(i)
		# print(mask)
		time2 = time.time()-t
		t = time.time()
		# convert pixel values (e.g. center)
		scaled_image = mold_image(image, cfg)
		time3 = time.time()-t
		t = time.time()		# convert image into one sample
		sample = expand_dims(scaled_image, 0)
		time4 = time.time()-t
		t = time.time()
		# make prediction
		yhat = model.detect(sample, verbose=0)[0]
		# print(yhat)
		time5 = time.time()-t
		t = time.time()

		'''Modificació'''
		folder = './Detection_RCNN_json/'
		if not os.path.exists(folder):
			os.mkdir(folder)
		else:
			pass

		name_json = f'{folder}RCNN_{i + 1}.json'

		tf = open(name_json, "w")
		json.dump(yhat, tf)
		tf.close()
		time6 = time.time() - t
		t = time.time()
		# data_json = json.dumps(str(yhat))
		# archivo_json = open(data_json, "w")
		# archivo_json.write(name_json)
		# archivo_json.close()

		# define subplot
		pyplot.subplot(1, 2, 1)
		# plot raw pixel data
		pyplot.imshow(image)
		pyplot.title('Actual')
		# plot masks
		for j in range(mask.shape[2]):
			pyplot.imshow(mask[:, :, j], cmap='gray', alpha=0.3)
		# get the context for drawing boxes
		pyplot.subplot(1, 2, 2)
		# plot raw pixel data
		pyplot.imshow(image)
		pyplot.title('Predicted')
		ax = pyplot.gca()
		# plot each box

		for box in yhat['rois']:
			# get coordinates
			y1, x1, y2, x2 = box
			# calculate width and height of the box
			width, height = x2 - x1, y2 - y1
			# create the shape
			rect = Rectangle((x1, y1), width, height, fill=False, color='red')
			# print(rect)
			# draw the box
			ax.add_patch(rect)

		# rect.remove()
		time7 = time.time() - t
		t = time.time()
		# show the figure

		'''Modificació'''
		folder = './Detection/'
		if not os.path.exists(folder):
			os.mkdir(folder)
		else:
			pass

		name_plot = f'{folder}img_num_{type_dataset}_{i + 1}'
		pyplot.savefig(name_plot)
		pyplot.close()
		time8 = time.time() - t
		t = time.time()
		ax.patches = []
		# pyplot.show(block=True)
		time9 = time.time() - t
		t = time.time()

		rprint(time1)
		rprint(time2)
		rprint(time3)
		rprint(time4)
		rprint(time5)
		rprint(time6)
		rprint(time7)
		rprint(time8)
		rprint(time9)


# load the train dataset
train_set = OBSEADataset()
train_set.load_dataset('OBSEA', is_train=True)
train_set.prepare()
print('Train: %d' % len(train_set.image_ids))
# load the test dataset
test_set = OBSEADataset()
test_set.load_dataset('OBSEA', is_train=False)
test_set.prepare()
rprint(test_set)
print('Test: %d' % len(test_set.image_ids))
# create config
cfg = PredictionConfig()
# define the model
model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
# load model weights
model_path = 'pesos200im_bo.h5'
model.load_weights(model_path, by_name=True)


# # plot predictions for train dataset
# plot_actual_vs_predicted(train_set, model, cfg)
# # plot predictions for test dataset
# plot_actual_vs_predicted(test_set, model, cfg)

'''Modificació'''
# plot predictions for train dataset
plot_actual_vs_predicted_2(train_set, model, cfg, type_dataset='train', n_images=len(train_set.image_ids))
# plot predictions for test dataset
plot_actual_vs_predicted_2(test_set, model, cfg, type_dataset='test', n_images=len(test_set.image_ids))

# plot_actual_vs_predicted(train_set, model, cfg)
# plot_actual_vs_predicted(test_set, model, cfg)

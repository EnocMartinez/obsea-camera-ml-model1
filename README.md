# Model Mask - RCNN  
## Object Detection with Mask R-CNN Keras model
  
The aim of this repository is to train a neural network capable of detecting fish from the image dataset of the underwater laboratory [OBSEA](https://www.obsea.es), linked to the research group [SARTI](https://www.sarti.webs.upc.edu/web_v2/).
  
The implemented code is based on the [Mask R-CNN repository for the detection and segmentation object](https://github.com/matterport/Mask_RCNN.git). Additional information can be found at [their website](https://machinelearningmastery.com/how-to-train-an-object-detection-model-with-keras/).

### Previous steps required
- Virtual environment with Python 3.6.
- Install requeriments.txt (principalment Keras == 2.2.4 & Tensor Flow == 1.15.3).

	```
	> pip install requeriments.txt
	```

- Install Mask R-CNN for Keras:
	1.  Git Clone [Mask R-CNN GitHub Repository](https://github.com/matterport/Mask_RCNN.git).
		
		``` 
		> git clone https://github.com/matterport/Mask_RCNN.git
		```
	3. Install Mask R-CNN Library
		
		``` 
		> cd Mask_RCNN
		> python setup.py install
		```
		
	4. Confirm the Library Was Installed
		``` 
		> pip show mask-rcnn
		```
- Download the weigths file of Mask R-CNN model [here](https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5).
- Install [Dataset OBSEA](https://github.com/uripratt/OBSEA-dataset/tree/master).
	``` { py }
	> git clone https://github.com/uripratt/OBSEA-dataset/tree/master
	```

	```
	OBSEA
	├── Tagging_XML
	└── Tagging_Img

	```

### Train Model
Change the "STEPS_PER_EPOCH" parameter if you will change the original DataSet:
Depending on the total number of photos in the dataset, it is necessary to separate approximately 90% for training and 10% to validation the model. In the present case a total of 3366 photos are trained. 


```
STEPS_PER_EPOCH = 3366
class OBSEADataset(Dataset):  
	   # load the dataset definitions  
	  def load_dataset(self, dataset_dir, is_train=True):  
	      # define one class  
	  self.add_class("dataset", 1, "OBSEA")  
	      # define data locations  
	  images_dir = dataset_dir + '/Tagging_Img/'  
	  annotations_dir = dataset_dir + '/Tagging_XML/'  
	  # find all images  
	  for filename in listdir(images_dir):  
	         # extract image id  
	  image_id = filename[:-4]  
	         # skip bad images  
	  if image_id in ['00090']:  
	            continue  
	  # skip all images after 3368 if we are building the train set  
	  if is_train and int(image_id) >= STEPS_PER_EPOCH:  
	            continue  
	  # skip all images before 3368 if we are building the test/val set  
	  if not is_train and int(image_id) < STEPS_PER_EPOCH:  
	            continue  
	  img_path = images_dir + filename  
	         ann_path = annotations_dir + image_id + '.xml'  
	  # add to dataset  
	  self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)
  ```

   ```
class OBSEAConfig(Config):  
		# define the name of the configuration  
		NAME = "OBSEA_cfg"  
		# number of classes (background + OBSEA)  
		NUM_CLASSES = 1 + 1  
		# number of training steps per epoch
		# STEPS_PER_EPOCH = number of training photos  
		STEPS_PER_EPOCH = STEPS_PER_EPOCH
 ```

Execute train_R_CNN_OBSEA.py

### Detect Objects

To detect objects it is necessary to run the Detect_OBSEA_RCNN.py code, in this case a dataset with which the model has not been trained is used. In this case this dataset has been used.

 
[<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/9/97/Logo_UPC.svg/1200px-Logo_UPC.svg.png" alt="drawing" width="200"/>](https://www.upc.edu/ca)	
[<img src="https://www.sarti.webs.upc.edu/web_v2/assets/onepage/img/logo/logo-obsea-medusa.png" alt="drawing" width="200"/>](https://www.obsea.es=) 
[<img src="https://www.sarti.webs.upc.edu/web_v2/assets/onepage/img/logo/logosarti.png" alt="drawing" width="600"/>](https://www.sarti.webs.upc.edu/web_v2/)

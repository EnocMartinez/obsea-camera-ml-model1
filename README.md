# Model Mask - RCNN  
## Object Detection with Mask R-CNN Keras model
  
The aim of this repository is to train a neural network capable of detecting fish from the image dataset of the underwater laboratory [OBSEA](https://www.obsea.es), linked to the research group [SARTI](https://www.sarti.webs.upc.edu/web_v2/).
  
The implemented code is based on the [Mask R-CNN repository for the detection and segmentation object](https://github.com/matterport/Mask_RCNN.git). Additional information can be found at [their website](https://machinelearningmastery.com/how-to-train-an-object-detection-model-with-keras/).

### Previous steps required
- #### [Virtual environment](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#creating-a-virtual-environment) with Python 3.6.
	- Installing virtualenv
		```
		py -m pip install --user virtualenv
		```
	- Creating a virtual environment
		```
		py -m venv /path/to/env
		```
	- Activating a virtual environment
		```
		.\env\Scripts\activate
		```

- #### Install requirements.txt (Keras == 2.2.4 & Tensor Flow == 1.15.3).
	```
	pip install -r /path/to/requirements.txt
	```


- #### Install Mask R-CNN for Keras:
	1.  Git Clone [Mask R-CNN GitHub Repository](https://github.com/matterport/Mask_RCNN.git).
		
		``` 
		> git clone https://github.com/matterport/Mask_RCNN.git
		```
	3. Install Mask R-CNN Library
		
		``` 
		> cd Mask_RCNN
		> python setup.py install
		```
		***S'ha de moure la carpeta mrcnn, que esta dinas de la carpeta intalada mask_rcnn, a l'entorn virtual.***
	4. Confirm the Library Was Installed
		``` 
		> pip show mask-rcnn
		```
- #### Download the weights file of Mask R-CNN model [here](https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5).
- #### Install [Dataset OBSEA](https://github.com/uripratt/OBSEA-dataset/tree/master).
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
```

Execute train_R_CNN_OBSEA.py

### Detect Objects

To detect objects it is necessary to run the code Detect_OBSEA_RCNN.py, in this case it has been trained with 10% of the images explained above (also used to validate the model). Therefore, the "STEPS_PER_EPOCH" paramater has to have the same value.

If the user wants to detect with another dataset, the following parameters must be changed:

***Explicar més endevant***


 
[<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/9/97/Logo_UPC.svg/1200px-Logo_UPC.svg.png" alt="drawing" width="200"/>](https://www.upc.edu/ca)	
[<img src="https://www.sarti.webs.upc.edu/web_v2/assets/onepage/img/logo/logo-obsea-medusa.png" alt="drawing" width="200"/>](https://www.obsea.es=) 
[<img src="https://www.sarti.webs.upc.edu/web_v2/assets/onepage/img/logo/logosarti.png" alt="drawing" width="600"/>](https://www.sarti.webs.upc.edu/web_v2/)
# Model Mask - RCNN  
## Object Detection with Mask R-CNN Keras model
  
The aim of this repository is to train a neural network capable of detecting fish from the image dataset of the underwater laboratory [OBSEA](https://www.obsea.es), linked to the research group [SARTI](https://www.sarti.webs.upc.edu/web_v2/).
  
The implemented code is based on the [Mask R-CNN repository for the detection and segmentation object](https://github.com/matterport/Mask_RCNN.git). Additional information can be found at [their website](https://machinelearningmastery.com/how-to-train-an-object-detection-model-with-keras/).

Previous steps required:  
- Virtual environment with Python 3.6.
- Install requeriments.txt (principalment Keras == 2.2.4 & Tensor Flow == 1.15.3).

	```
	> pip install requeriments.txt
	```
  
- Install Mask R-CNN for Keras:
	1.  Git Clone [Mask R-CNN GitHub Repository](https://github.com/matterport/Mask_RCNN.git).
		
		``` { py }
		> git clone https://github.com/matterport/Mask_RCNN.git
		```
	3. Install Mask R-CNN Library
		
		``` { py }
		> cd Mask_RCNN
		> python setup.py install
		```
		
	4. Confirm the Library Was Installed
		``` { py }
		> pip show mask-rcnn
		```

- Install [Dataset OBSEA](https://github.com/uripratt/OBSEA-dataset/tree/master).
	``` { py }
	> git clone https://github.com/uripratt/OBSEA-dataset/tree/master
	```

	```
	OBSEA
	├── Tagging_XML
	└── Tagging_Img

	```


[<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/9/97/Logo_UPC.svg/1200px-Logo_UPC.svg.png" alt="drawing" width="200"/>](https://www.upc.edu/ca)	
[<img src="https://www.sarti.webs.upc.edu/web_v2/assets/onepage/img/logo/logo-obsea-medusa.png" alt="drawing" width="200"/>](https://www.obsea.es=) 
[<img src="https://www.sarti.webs.upc.edu/web_v2/assets/onepage/img/logo/logosarti.png" alt="drawing" width="600"/>](https://www.sarti.webs.upc.edu/web_v2/)

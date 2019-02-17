# Object detection module Installation Guide

	conda create -n retinanet python=3.6 anaconda
#Create an Anaconda environment with python version 3.6.

	source activate retinanet
	conda install tensorflow numpy scipy opencv pillow matplotlib h5py keras
#Install necessary packages

	pip3 install --upgrade tensorflow --user
#update tensorflow
	
	pip3 install https://github.com/OlafenwaMoses/ImageAI/releases/download/2.0.2/imageai-2.0.2-py3-none-any.whl --user
#Install the ImageAI library

	wget https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/resnet50_coco_best_v2.0.1.h5
#download the pretrained model required to generate predictions into your current working folder. This model is based on RetinaNet

	wget https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2018/06/I1_2009_09_08_drive_0012_001351-768x223.png
#download and copy the image to your working directory. Rename the image to image.png

	jupyter notebook
#open jupyter notebook in the terminal and run the following code

	>>> from imageai.Detection import ObjectDetection
	>>> import os

	>>> execution_path = os.getcwd()

	>>> detector = ObjectDetection()
	>>> detector.setModelTypeAsRetinaNet()
	>>> detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
	>>> detector.loadModel()
	>>> custom_objects = detector.CustomObjects(person=True, car=False)
	>>> detections = detector.detectCustomObjectsFromImage(input_image=os.path.join(execution_path , "image.png"), output_image_path=os.path.join(execution_path , "image_new.png"), custom_objects=custom_objects, minimum_percentage_probability=65)

	>>> for eachObject in detections:
   	>>> print(eachObject["name"] + " : " + eachObject["percentage_probability"] )
   	>>> print("--------------------------------")

# this will create a modifed image file name image_new.png which contains the bounding box for your image

	>>> from IPython.display import Image
	>>> Image("image_new.png")

![](<a href="https://ibb.co/bvCM7qC"><img src="https://i.ibb.co/fNyzQBy/output-zl-Odx-A.gif" alt="output-zl-Odx-A" border="0"></a><br /><a target='_blank' href='https://aluminumsulfate.net/ammonium-sulfate'>nh4 2so4 formula name</a><br />)

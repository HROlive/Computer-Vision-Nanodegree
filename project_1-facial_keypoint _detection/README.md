# Facial Keypoint Detection

## Project Overview

In this project, you’ll combine your knowledge of computer vision techniques and deep learning architectures to build a facial keypoint detection system. Facial keypoints include points around the eyes, nose, and mouth on a face and are used in many applications. These applications include: facial tracking, facial pose recognition, facial filters, and emotion recognition. Your completed code should be able to look at any image, detect faces, and predict the locations of facial keypoints on each face; examples of these keypoints are displayed below.

<p align="center">
  <img src="/images/key_pts_example.png">
</p>

The project will be broken up into a few main parts in four Python notebooks:

[1. Load and Visualize Data](https://github.com/HROlive/Computer-Vision-Nanodegree/blob/master/project_1-facial_keypoint%20_detection/1.%20Load%20and%20Visualize%20Data.ipynb) : Loading and Visualizing the Facial Keypoint Data

[2. Define the Network Architecture](https://github.com/HROlive/Computer-Vision-Nanodegree/blob/master/project_1-facial_keypoint%20_detection/2.%20Define%20the%20Network%20Architecture.ipynb) : Defining and Training a Convolutional Neural Network (CNN) to Predict Facial Keypoints

[3. Facial Keypoint Detection, Complete Pipeline](https://github.com/HROlive/Computer-Vision-Nanodegree/blob/master/project_1-facial_keypoint%20_detection/3.%20Facial%20Keypoint%20Detection%2C%20Complete%20Pipeline.ipynb) : Facial Keypoint Detection Using Haar Cascades and your Trained CNN

[4. Fun with Keypoints](https://github.com/HROlive/Computer-Vision-Nanodegree/blob/master/project_1-facial_keypoint%20_detection/4.%20Fun%20with%20Keypoints.ipynb) : Fun Filters and Keypoint Uses

<p align="center">
  <img src="images/facial_keypoint_inference.jpg">
</p>

## Project Instructions

All of the starting code and resources you'll need to compete this project are in this Github repository. Before you can get started coding, you'll have to make sure that you have all the libraries and dependencies required to support this project. If you have already created a `cv-nd` environment for [exercise code](https://github.com/udacity/CVND_Exercises), then you can use that environment! If not, instructions for creation and activation are below.


### Local Environment Instructions

1. Clone the repository, and navigate to the downloaded folder. This may take a minute or two to clone due to the included image data.
```
git clone https://github.com/udacity/P1_Facial_Keypoints.git
cd P1_Facial_Keypoints
```

2. Create (and activate) a new environment, named `cv-nd` with Python 3.6. If prompted to proceed with the install `(Proceed [y]/n)` type y.

	- __Linux__ or __Mac__: 
	```
	conda create -n cv-nd python=3.6
	source activate cv-nd
	```
	- __Windows__: 
	```
	conda create --name cv-nd python=3.6
	activate cv-nd
	```
	
	At this point your command line should look something like: `(cv-nd) <User>:P1_Facial_Keypoints <user>$`. The `(cv-nd)` indicates that your environment has been activated, and you can proceed with further package installations.

3. Install PyTorch and torchvision; this should install the latest version of PyTorch.
	
	- __Linux__ or __Mac__: 
	```
	conda install pytorch torchvision -c pytorch 
	```
	- __Windows__: 
	```
	conda install pytorch-cpu -c pytorch
	pip install torchvision
	```

6. Install a few required pip packages, which are specified in the requirements text file (including OpenCV).
```
pip install -r requirements.txt
```


### Data

All of the data you'll need to train a neural network is in the P1_Facial_Keypoints repo, in the subdirectory `data`. In this folder are training and tests set of image/keypoint data, and their respective csv files. This will be further explored in Notebook 1: Loading and Visualizing Data, and you're encouraged to look trough these folders on your own, too.


## Notebooks

1. Navigate back to the repo. (Also, your source environment should still be activated at this point.)
```shell
cd
cd P1_Facial_Keypoints
```

2. Open the directory of notebooks, using the below command. You'll see all of the project files appear in your local environment; open the first notebook and follow the instructions.
```shell
jupyter notebook
```

3. Once you open any of the project notebooks, make sure you are in the correct `cv-nd` environment by clicking `Kernel > Change Kernel > cv-nd`.

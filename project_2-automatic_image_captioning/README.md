# Automatic Image Captioning

## Project Objective

In this project we combine Convolutional Neural Networks (CNN) and Recurrent Neural Networks (RNN) knowledge to build a deep learning model that produces captions given an input image.

Image captioning requires that you create a complex deep learning model with two components: a CNN that transforms an input image into a set of features, and an RNN that turns those features into rich, descriptive language.

One such example of how this architecture performs is pictured below:

<p align="center">
  <img src="images/image-description.png">
</p>

## Getting the Files

### Files Description

* `0_Dataset.ipynb`: Explore MS COCO dataset using COCO API
* `1_Preliminaries.ipynb`: Load and pre-process data from the MS COCO dataset and design the CNN-RNN model for automatically generating image captions
* `2_Training.ipynb`: Training phase of the CNN-RNN model 
* `3_Inference.ipynb`: Using the previously trained model to generate captions for images in the test dataset.
* `model.py` : File containing the model architecture
* `data_loader.py` : Custom data loader for PyTorch combining the dataset and the sampler
* `vocabulary.py` : Vocabulary constructor built from the captions in the training dataset
* `vocab.pkl` : Vocabulary file stored to load it immediately from the data loader
* `training_log.txt` : Detailed log of the training phase

### Data

The Microsoft **C**ommon **O**bjects in **CO**ntext (MS COCO) dataset is a large-scale dataset for scene understanding.  The dataset is commonly used to train and benchmark object detection, segmentation, and captioning algorithms.  

<p align="center">
  <img src="images/images/coco-examples.jpg">
</p>

You can read more about the dataset on the [website](http://cocodataset.org/#home) or in the [research paper](https://arxiv.org/pdf/1405.0312.pdf).

To obtain and explore the dataset, you can use either the [COCO API](https://github.com/cocodataset/cocoapi), or run the [Dataset notebook](0_Dataset.ipynb).

### Model

By merging the CNN encoder and the RNN decoder, we can get a model that can find patterns in images and then use that information to help generate a description of those images. The input image will be processed by a CNN and we will connect the output of the CNN to the input of the RNN which will allow us to generate descriptive text.

<p align="center">
  <img src="images/images/images/encoder-decoder.png">
</p>

Please feel free to experiment with alternative architectures, such as bidirectional LSTM with attention mechanisms.

## Result

The result should be an inference like the one sampled bellow:

![Caption example](images/caption_inference.jpg)

## Dependencies

Before you can experiment with the code, you'll have to make sure that you have all the libraries and dependencies required to support this project. You will mainly need Python 3, PyTorch and its torchvision, OpenCV, Matplotlib. You can install many dependencies using `pip3 install -r requirements.txt`.

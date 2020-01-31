# Using Mask R-CNN to detect Car Damage
Using the amazing Matterport's [Mask_RCNN](https://github.com/nicolasmetallo/Mask_RCNN) implementation and following Priya's [example](https://www.analyticsvidhya.com/blog/2018/07/building-mask-r-cnn-model-detecting-damage-cars-python/), I trained an algorithm that highlights areas where there is damage to a car (i.e. dents, scratches, etc.). You can run the step-by-step notebook in Google Colab or use the following:
```
Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 custom.py train --dataset=/path/to/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 custom.py train --dataset=/path/to/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 custom.py train --dataset=/path/to/dataset --weights=imagenet

    # Apply color splash to an image
    python3 custom.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 custom.py splash --weights=last --video=<URL or path to file>
"""
```
## Installation
This script supports Python 2.7 and 3.7, although if you run into problems with TensorFlow and Python 3.7, it might be easier to just run everything from Google Colaboratory notebook.

## Clone this repository

## Install pre-requisites
```
$ pip install -r requirements.txt
```

## Train your model
This step is done only first time.
We download the 'coco' weights and start training from that point with our custom images (transfer learning). It will download the weights automatically if it can't find them. Training takes around 4 mins per epoch with 10 epochs.
```
$ python3 custom.py train --dataset='dataset' --weights=coco # it will download coco weights if you don't have them
```

## Start the server
Weights of a pre trained model is already present in the logs folder. Run the index.py file

## Check for damage in the image
Send any car image file using postman as form-data as a POST request at http://localhost:5005/api


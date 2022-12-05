# Authors:

# Description: Use deep learning principles on a dataset of your choosing. We chose a dataset of citrus leaves with afflictions,
# as well as healthy leaves for comparison. We will be using these images to predict which conditions (greening, black spot, canker, healthy)
# the leaf appears to be. Following that, we will evaluate the performance of our algorithm implementation in order to provide metrics by which
# out project may be judged and improved.

# Dataset is available from the following link:
# https://www.tensorflow.org/datasets/catalog/citrus_leaves

import glob #used to get a list of all the files in a folder
from cv2 import imread, imwrite, resize, IMREAD_UNCHANGED #select the tools we need process our images

images = list()
black_spot_resized = list()
canker_resized = list()
greening_resized = list()
healthy_resized = list()
melanose_resized = list()

#Shrink image aspect ratio to improve runtime, test multiple epochs to find satisfactory metrics

new_dimension = 8 #we can just treat images as squared, setting their new pixel dimensions using a single variable

#Get a list of the images in the given directory, then for each of those images, make a temporary file of that image with a new aspect ratio
images = glob.glob('./Citrus/Leaves/Black spot/*.png') #get the names files in the Black spot folder as a list

for image in images:
    source_image = imread(image, IMREAD_UNCHANGED) #set the source image
    dimension_size = (new_dimension, new_dimension) #set dimensions_size to be new_dimension pixels in height and width
    resized_image = resize(source_image, dimension_size) #change the dimensions of the image
    black_spot_resized.append(resized_image) #add the new temporary images to a list


    





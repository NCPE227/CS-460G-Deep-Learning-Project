# Authors:

# Description: Use deep learning principles on a dataset of your choosing. We chose a dataset of citrus leaves with afflictions,
# as well as healthy leaves for comparison. We will be using these images to predict which conditions (greening, black spot, canker, healthy)
# the leaf appears to be. Following that, we will evaluate the performance of our algorithm implementation in order to provide metrics by which
# out project may be judged and improved.

# Dataset is available from the following link:
# https://www.tensorflow.org/datasets/catalog/citrus_leaves

import glob # used to get a list of all the files in a folder
from cv2 import imread, resize, IMREAD_UNCHANGED # select the tools we need process our images

# Lists to use for the output of the temporarily resized images
black_spot = list()
canker = list()
greening = list()
healthy = list()
melanose = list()

master_images = list() #compiled after resizing to contain images from all sets

# We meed to shrink image aspect ratio to improve runtime, and will test multiple epochs to find satisfactory metrics

# When this function is run, it resizes all 5 image sets to a given square aspect ratio, doesn't write the images, but keeps them in a list
def Resize_Images(new_dimension):
    images = list()
    black_spot_resized = list()
    canker_resized = list()
    greening_resized = list()
    healthy_resized = list()
    melanose_resized = list()

    dimension_size = (new_dimension, new_dimension) #set dimensions_size to be new_dimension pixels in height and width

    #Get a list of the images in the given directory, then for each of those images, make a temporary file of that image with a new aspect ratio
    images = glob.glob('./Citrus/Leaves/Black spot/*.png') #get the names files in the Black spot folder as a list

    for image in images: #resize black spot images
        image = image.replace('\\', '/')
        source_image = imread(image, IMREAD_UNCHANGED) #set the source image
        #print('Original Dimension: ', source_image.shape)
        resized_image = resize(source_image, dimension_size) #change the dimensions of the image
        #print('New Dimension: ', source_image.shape)
        black_spot_resized.append(resized_image) #add the new temporary images to a list
    #print('Size of Black Spot List: ', len(black_spot_resized))

    #Get a list of the images in the given directory, then for each of those images, make a temporary file of that image with a new aspect ratio
    images = glob.glob('./Citrus/Leaves/canker/*.png') #get the names files in the Black spot folder as a list
 
    for image in images: #resize canker images
        image = image.replace('\\', '/')
        source_image = imread(image, IMREAD_UNCHANGED) #set the source image
        #print(source_image)
        #print('Original Dimension: ', source_image.shape)
        resized_image = resize(source_image, dimension_size) #change the dimensions of the image
        #print('New Dimension: ', resized_image.shape)
        canker_resized.append(resized_image) #add the new temporary images to a list
    #print('Size of canker List: ', len(black_spot_resized))

    #Get a list of the images in the given directory, then for each of those images, make a temporary file of that image with a new aspect ratio
    images = glob.glob('./Citrus/Leaves/greening/*.png') #get the names files in the Black spot folder as a list

    for image in images: #resize greening images
        image = image.replace('\\', '/')
        source_image = imread(image, IMREAD_UNCHANGED) #set the source image
        #print('Original Dimension: ', source_image.shape)
        resized_image = resize(source_image, dimension_size) #change the dimensions of the image
        greening_resized.append(resized_image) #add the new temporary images to a list

    #Get a list of the images in the given directory, then for each of those images, make a temporary file of that image with a new aspect ratio
    images = glob.glob('./Citrus/Leaves/healthy/*.png') #get the names files in the Black spot folder as a list

    for image in images: #resize healthy images
        image = image.replace('\\', '/')
        source_image = imread(image, IMREAD_UNCHANGED) #set the source image
        #print('Original Dimension: ', source_image.shape)
        resized_image = resize(source_image, dimension_size) #change the dimensions of the image
        healthy_resized.append(resized_image) #add the new temporary images to a list

    #Get a list of the images in the given directory, then for each of those images, make a temporary file of that image with a new aspect ratio
    images = glob.glob('./Citrus/Leaves/Melanose/*.png') #get the names files in the Black spot folder as a list

    for image in images: #resize melanose images
        image = image.replace('\\', '/')
        source_image = imread(image, IMREAD_UNCHANGED) #set the source image
        #print('Original Dimension: ', source_image.shape)
        resized_image = resize(source_image, dimension_size) #change the dimensions of the image
        melanose_resized.append(resized_image) #add the new temporary images to a list

    return black_spot_resized, canker_resized, greening_resized, healthy_resized, melanose_resized #return all the lists

# OpenCV (cv2) uses BGR rather than RGB orientation
# Use the BGR values of the pixels in our images post-scaling to determine the color of each pixel, then get a ratio of the colors of each image
# which we will use as our basis for predicting which condition the image depicts.
def BGR_Calculation(image_list, new_dimension):
    x_dimension = new_dimension
    y_dimension = new_dimension
    pixel_colors = list() # temporary list for each pixel
    image_colors = list() # list of lists, each sublist contains the color predictions for the pixels of the given image

    #For each image in the image_list, we are going to get the BGR values of each pixel in the image
    for image in image_list:
        pixel_colors = [] #set the temp list back to empty

        # Calculate the BGR values for each pixel in the image
        for pixel_y in y_dimension:
            for pixel_x in x_dimension:
                pixel_colors.append(image[pixel_x, pixel_y])
        
        # needs if/else logic here to determine the color of a pixel based on a range of values in the BGR slots,
        # from there, we just need to figure out an average color ratio for each class, which will be used to predict our values later
        # once we have that, we should be able to validate and predict to our heart's content.
        
        # Now we need to decide what color each of those pixels are     


    return

# This function creates the master list of all the datasets
def Combine():
    master_images.append(black_spot)
    master_images.append(canker)
    master_images.append(greening)
    master_images.append(healthy)
    master_images.append(melanose)

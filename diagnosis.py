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

master_images = list() # compiled after resizing to contain images from all sets
master_colors = list() # same as master images but each pixel is a string name for its color instead of a BGR component list
master_file_names = list() # contains the names of the files in their appropriate order
true_class = list() # a list of the true classifications of each image

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
#
# Returns a list of the average occurence of colors per each condition. We will use this for predictions
def BGR_Calculation(image_list, new_dimension):
    x_dimension = new_dimension
    y_dimension = new_dimension
    pixel_colors = list() # temporary list for each pixel
    image_colors = list() # list of lists, each sublist contains the color predictions for the pixels of the given image
    color_name = list() # list of color names that correspond to each pixel of each image

    # For each image in the image_list, we are going to get the BGR values of each pixel in the image
    for image in image_list:
        pixel_colors = [] #set the temp list back to empty

        # Calculate the BGR values for each pixel in the image
        for pixel_y in range(y_dimension):
            for pixel_x in range(x_dimension):
                pixel_colors.append(image[pixel_x, pixel_y].tolist())
        
        image_colors.append(pixel_colors) # add the list of arrays to a master list where each of these lists is an image
    
    # For each pixel in each image in image_colors, determine it's color by the BGR ratio
    for image in image_colors:
        temp_color_name = list() # use this to create a list for an individual image, then add the list to the color_name list
        for pixel in image:
            if ((pixel[2] > pixel[0]) & (pixel[2] > pixel[1])): # detect a brown pixel
                temp_color_name.append('brown')
            elif ((pixel[0] > pixel[1]) & (pixel[0] > pixel[2])): # detect the background of an image
                temp_color_name.append('background')
            elif ((pixel[1] > pixel[0]) & (pixel[1] > pixel[2])): # detect a green pixel
                temp_color_name.append('green')
            elif ((pixel[0] > pixel[2]) & (pixel[1] > pixel[2])): # detect a yellow pixel
                temp_color_name.append('yellow')
        color_name.append(temp_color_name) # add the temp list to this so we have a list of lists which corresponds with the pixels in the image_colors list of lists
    #print(color_name)

    # Count the appearances of each of the leaf colors, ignoring the background colors
    green, brown, yellow = 0, 0, 0 # color counting variables
    average_green, average_yellow, average_brown = 0, 0, 0
    for image in color_name:
        for color in image:
            #print(color)
            if color == 'green':
                green += 1
            elif color == 'brown':
                brown += 1
            elif color == 'yellow':
                yellow +=1
        # Add the counted values for this image to the average variable
        average_green += green
        average_brown += brown
        average_yellow += yellow
    
    # Divide each category by the number of items in color_name to get an average color value for each class
    average_green = average_green / len(color_name)
    average_brown = average_brown / len(color_name)
    average_yellow = average_yellow / len(color_name)

    average_colors = [average_green, average_brown, average_yellow] # create a variable to store the averages to be returned
            
    return average_colors, color_name

# This function creates the master list of all the datasets and a list of their true classifications
def Combine_Lists():
    for each in range(len(black_spot)):
        master_images.append(black_spot[each])
        master_colors.append(black_spot_pixels_as_color[each])
        master_file_names.append(f'b{each}.png')
        true_class.append('Black Spot')
    
    for each in range(len(canker)):
        master_images.append(canker[each])
        master_colors.append(canker_pixels_as_color[each])
        master_file_names.append(f'c{each}.png')
        true_class.append('Canker')
    
    for each in range(len(greening)):
        master_images.append(greening[each])
        master_colors.append(greening_pixels_as_color[each])
        master_file_names.append(f'g{each}.png')
        true_class.append('Greening')
    
    for each in range(len(healthy)):
        master_images.append(healthy[each])
        master_colors.append(healthy_pixels_as_color[each])
        master_file_names.append(f'h{each}.png')
        true_class.append('Healthy')
    
    for each in range(len(melanose)):
        master_images.append(melanose[each])
        master_colors.append(melanose_pixels_as_color[each])
        master_file_names.append(f'm{each}.png')
        true_class.append('Melanose')

dimension = 4 # lowered for quicker testing 

black_spot, canker, greening, healthy, melanose = Resize_Images(dimension) # Resize images and set each of them to be compatible with OpenCV

# Get the average representation of each of our identifier colors from the datasets, output is a list representing the average
# presence of that color for the entire class represented by the list. This is in the form of [ green, brown, yellow ]. It also returns
# a list of each image where all the pixels have been converted to their color names
black_spot_color_averages, black_spot_pixels_as_color = BGR_Calculation(black_spot, dimension)
canker_color_averages, canker_pixels_as_color = BGR_Calculation(canker, dimension)
greening_color_averages, greening_pixels_as_color = BGR_Calculation(greening, dimension)
healthy_color_averages, healthy_pixels_as_color = BGR_Calculation(healthy, dimension)
melanose_color_averages, melanose_pixels_as_color = BGR_Calculation(melanose, dimension)

# Combine lists to make a master of the previous datasets in order to split into training and testing data.
Combine_Lists()
#print(black_spot[0])

#Simply for proof of run, this makes it take a LONG TIME, leave commented out for real runs.
#print(master_images[0])
#print()
#print(master_images[1])
#print(master_colors[0])
#print(len(master_colors[0]))

'''#Testing using a single image
source_image = imread('./Citrus/Leaves/Black spot/b0.png', IMREAD_UNCHANGED) #set the source image
#print('Original Dimension: ', source_image.shape)
resized_image = resize(source_image, (256, 256)) #change the dimensions of the image
#print('New Dimension: ', source_image.shape)
black_spot.append(resized_image) #add the new temporary images to a list
#print('Size of Black Spot List: ', len(black_spot_resized))
average_occurences = BGR_Calculation(black_spot, 256)
print('Black Spot Average: ', average_occurences)

source_image = imread('./Citrus/Leaves/canker/c0.png', IMREAD_UNCHANGED) #set the source image
#print('Original Dimension: ', source_image.shape)
resized_image = resize(source_image, (256, 256)) #change the dimensions of the image
#print('New Dimension: ', source_image.shape)
canker.append(resized_image) #add the new temporary images to a list
#print('Size of Black Spot List: ', len(black_spot_resized))
average_occurences = BGR_Calculation(canker, 256)
print('Canker Average: ', average_occurences)

source_image = imread('./Citrus/Leaves/greening/g0.png', IMREAD_UNCHANGED) #set the source image
#print('Original Dimension: ', source_image.shape)
resized_image = resize(source_image, (256, 256)) #change the dimensions of the image
#print('New Dimension: ', source_image.shape)
greening.append(resized_image) #add the new temporary images to a list
#print('Size of Black Spot List: ', len(black_spot_resized))
average_occurences = BGR_Calculation(greening, 256)
print('Greening Average: ', average_occurences)

source_image = imread('./Citrus/Leaves/healthy/h0.png', IMREAD_UNCHANGED) #set the source image
#print('Original Dimension: ', source_image.shape)
resized_image = resize(source_image, (256, 256)) #change the dimensions of the image
#print('New Dimension: ', source_image.shape)
healthy.append(resized_image) #add the new temporary images to a list
#print('Size of Black Spot List: ', len(black_spot_resized))
average_occurences = BGR_Calculation(healthy, 256)
print('Healthy Average: ', average_occurences)

source_image = imread('./Citrus/Leaves/Melanose/m0.png', IMREAD_UNCHANGED) #set the source image
#print('Original Dimension: ', source_image.shape)
resized_image = resize(source_image, (256, 256)) #change the dimensions of the image
#print('New Dimension: ', source_image.shape)
melanose.append(resized_image) #add the new temporary images to a list
#print('Size of Black Spot List: ', len(black_spot_resized))
average_occurences = BGR_Calculation(melanose, 256)
print('Melanose Average: ', average_occurences)'''



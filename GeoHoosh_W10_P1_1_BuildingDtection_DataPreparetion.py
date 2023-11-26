# -*- coding: Building detection using U-Net from UHR Remote Sensing Images -*-
"""
Created on Sun Oct  1 17:30:43 2023

@author: Ehsan Khankeshizadeh

Tasks achieved:
1. Read large images and corresponding masks, divide them into smaller patches.
And write the patches as images to the local drive.  

2. Save only images and masks where masks have some decent amount of labels other than 0. 
Using blank images with label=0 is a waste of time and may bias the model towards 
unlabeled pixels. 

3. Divide the sorted dataset from above into train and validation datasets. 

"""
#Load Libraries
import os
import cv2
import numpy as np
import glob
import numpy as np
from matplotlib import pyplot as plt
from patchify import patchify
import tifffile as tiff
from PIL import Image
import random

#Change directory
os.chdir('E:\\ExteraWork\\GeoHoosh\\Section_Ehsan\\W10\\P1_BuildingDetection')
from Functions import Patching_img, Patching_mask, Realinformation

#Quick understanding of the dataset
temp_img = cv2.imread("Datasets/image/img_1.tif") #3 channels / spectral bands
plt.imshow(temp_img[:,:,[2,1,0]]) #View each channel...
temp_mask = cv2.imread("Datasets/mask/img_1.tif") #3 channels but all same. 
labels, count = np.unique(temp_mask[:,:,0], return_counts=True) #Check for each channel. All chanels are identical
print("Labels are: ", labels, " and the counts are: ", count)
#____________________________________________________________

#Now, crop each large image into patches of 256x256. Save them into a directory 
#so we can use data augmentation and read directly from the drive. 
#Creat new folder for save patched data in directory
newpath = root_directory+'256_patches/images'
if not os.path.exists(newpath):
    os.makedirs(newpath)

newpath2 = root_directory+'256_patches/masks'
if not os.path.exists(newpath2):
    os.makedirs(newpath2)

            
 #Now do the same as above for masks
 #For this specific dataset we could have added masks to the above code as masks have extension png
root_directory = 'Datasets/'
patch_size = 512
img_dir=root_directory+"image/"
mask_dir=root_directory+"mask/"

Patching_img(root_directory,img_dir,patch_size)
Patching_mask(root_directory,mask_dir,patch_size)

train_img_dir = "Datasets/256_patches/images/"
train_mask_dir = "Datasets/256_patches/masks/"

img_list = os.listdir(train_img_dir)
msk_list = os.listdir(train_mask_dir)

num_images = len(os.listdir(train_img_dir))


img_num = random.randint(0, num_images-1)

img_for_plot = cv2.imread(train_img_dir+img_list[img_num], 1)
img_for_plot = cv2.cvtColor(img_for_plot, cv2.COLOR_BGR2RGB)

mask_for_plot =cv2.imread(train_mask_dir+msk_list[img_num], 0)

plt.figure(figsize=(12, 8))
plt.subplot(121)
plt.imshow(img_for_plot)
plt.title('Image')
plt.subplot(122)
plt.imshow(mask_for_plot, cmap='gray')
plt.title('Mask')
plt.show()
#____________________________________________________________

#Now, let us copy images and masks with real information to a new folder.
# real information = if mask has decent amount of labels other than 0. 

#Creat new folder for save real information in directory
newpath3 = root_directory+'images_with_useful_info/images'
if not os.path.exists(newpath3):
    os.makedirs(newpath3)

newpath4 = root_directory+'images_with_useful_info/masks'
if not os.path.exists(newpath4):
    os.makedirs(newpath4)

Realinformation(img_list,msk_list,train_img_dir,train_mask_dir)
#____________________________________________________________
#Now split the data into training and validation
"""
Code for splitting folder into train, test, and val.
Once the new folders are created rename them and arrange in the format below to be used
for semantic segmentation using data generators. 


pip install split-folders
"""
#Creat new folder for save splited train and validation datasets
newpath5 = root_directory+'data_for_training_and_testing/'
if not os.path.exists(newpath5):
    os.makedirs(newpath5)

import splitfolders  # or import split_folders

input_folder = root_directory+ '/images_with_useful_info/'
output_folder =root_directory+'/data_for_training_and_testing/'

# Split with a ratio:
# To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(.75, .25), group_prefix=None) # default values
#____________________________________________________________

















 
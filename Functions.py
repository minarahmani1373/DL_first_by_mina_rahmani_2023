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

#Read images from repsective 'images' subdirectory
#As all images are of different size we have 2 options, either resize or crop
#But, some images are too large and some small. Resizing will change the size of real objects.
#Therefore, we will crop them to a nearest size divisible by 256 and then 
#divide all images into patches of 256x256x3. 

def Patching_img(root_directory,img_dir,patch_size):
    for path, subdirs, files in os.walk(img_dir):
        #pr int(path)  
        dirname = path.split(os.path.sep)[-1]
        #print(dirname)
        images = os.listdir(path)  #List of all image names in this subdirectory
        #print(images)
        for i, image_name in enumerate(images):  
            if image_name.endswith(".tif"):
                #print(image_name)
                image = cv2.imread(path+"/"+image_name, 1)  #Read each image as BGR
                SIZE_X = (image.shape[1]//patch_size)*patch_size #Nearest size divisible by our patch size
                SIZE_Y = (image.shape[0]//patch_size)*patch_size #Nearest size divisible by our patch size
                image = Image.fromarray(image)
                image = image.crop((0 ,0, SIZE_X, SIZE_Y))  #Crop from top left corner
                #image = image.resize((SIZE_X, SIZE_Y))  #Try not to resize for semantic segmentation
                image = np.array(image)             
       
                #Extract patches from each image
                print("Now patchifying image:", path+"/"+image_name)
                patches_img = patchify(image, (patch_size, patch_size, 3), step=patch_size)  #Step=256 for 256 patches means no overlap
        
                for i in range(patches_img.shape[0]):
                    for j in range(patches_img.shape[1]):
                        
                        single_patch_img = patches_img[i,j,:,:]
                        #single_patch_img = (single_patch_img.astype('float32')) / 255. #We will preprocess using one of the backbones
                        single_patch_img = single_patch_img[0] #Drop the extra unecessary dimension that patchify adds.                               
                        
                        cv2.imwrite(root_directory+"512_patches/images/"+
                                   image_name+"patch_"+str(i)+str(j)+".tif", single_patch_img)
                        #image_dataset.append(single_patch_img)
 
            
def Patching_mask(root_directory,mask_dir,patch_size):
    for path, subdirs, files in os.walk(mask_dir):
        #print(path)  
        dirname = path.split(os.path.sep)[-1]

        masks = os.listdir(path)  #List of all image names in this subdirectory
        for i, mask_name in enumerate(masks):  
            if mask_name.endswith(".tif"):           
                mask = cv2.imread(path+"/"+mask_name, 0)  #Read each image as Grey (or color but remember to map each color to an integer)
                SIZE_X = (mask.shape[1]//patch_size)*patch_size #Nearest size divisible by our patch size
                SIZE_Y = (mask.shape[0]//patch_size)*patch_size #Nearest size divisible by our patch size
                mask = Image.fromarray(mask)
                mask = mask.crop((0 ,0, SIZE_X, SIZE_Y))  #Crop from top left corner
                #mask = mask.resize((SIZE_X, SIZE_Y))  #Try not to resize for semantic segmentation
                mask = np.array(mask)             
       
                #Extract patches from each image
                print("Now patchifying mask:", path+"/"+mask_name)
                patches_mask = patchify(mask, (patch_size, patch_size), step=patch_size)  #Step=256 for 256 patches means no overlap
        
                for i in range(patches_mask.shape[0]):
                    for j in range(patches_mask.shape[1]):
                        
                        single_patch_mask = patches_mask[i,j,:,:]
                        #single_patch_img = (single_patch_img.astype('float32')) / 255. #No need to scale masks, but you can do it if you want
                        #single_patch_mask = single_patch_mask[0] #Drop the extra unecessary dimension that patchify adds.                               
                        cv2.imwrite(root_directory+"512_patches/masks/"+
                                   mask_name+"patch_"+str(i)+str(j)+".tif", single_patch_mask)
    
    
def Realinformation(img_list,msk_list,train_img_dir,train_mask_dir):
    useless=0  #Useless image counter
    for img in range(len(img_list)):   #Using t1_list as all lists are of same size
        img_name=img_list[img]
        mask_name = msk_list[img]
        print("Now preparing image and masks number: ", img)
          
        temp_image=cv2.imread(train_img_dir+img_name, 1)
       
        temp_mask=cv2.imread(train_mask_dir+mask_name, 0)
        #temp_mask=temp_mask.astype(np.uint8)
        
        val, counts = np.unique(temp_mask, return_counts=True)
        
        if (1 - (counts[0]/counts.sum())) > 0.05:  #At least 5% useful area with labels that are not 0
            print("Save Me")
            cv2.imwrite('Datasets/images_with_useful_info/images/'+img_name, temp_image)
            cv2.imwrite('Datasets/images_with_useful_info/masks/'+mask_name, temp_mask)
            
        else:
            print("I am useless")   
            useless +=1
    
    print("Total useful images are: ", len(img_list)-useless)  #20,075
    print("Total useless images are: ", useless) #21,571
    
 
 
    
 
    
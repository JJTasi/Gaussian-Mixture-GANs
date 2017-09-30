from os import listdir
from PIL import Image as PImage
from scipy.ndimage import imread
from skimage.transform import resize
import matplotlib.pyplot as plt
import numpy as np
import random

def rgb2gray(rgb):
    tmp=rgb.astype(np.float32)/255.
    tmp=np.reshape(tmp,(94*94,3))
    tmp=np.dot(tmp[:,:3], [0.299, 0.587, 0.114])
    return np.reshape(tmp,(94,94,1))

def loadImages(path):
    #imagesList = listdir(path)
    #print (imagesList)
    f=open("/home/jay/beta/DCGAN/img_list.txt")
    image_list=[]
    for img_name in f:
        img_path = path+img_name[:-1]
        tmp=np.zeros([128,128,3])
        if img_path[-3:] == 'png':
            img = imread(img_path)
            if img.shape != (100,100,3):
                img=resize(img,(100,100,3))
            tmp[14:114,14:114,:]=img
            image_list.append(img[3:97,3:97,:])
            #image_list.append(tmp)
            
    images_array=np.reshape(image_list,(len(image_list),94,94,3))
    index=range(images_array.shape[0])
    random.shuffle(index)
    return images_array[index,:,:,:]
    #return image_list
    
def loadImages_gray(path):
    #imagesList = listdir(path)
    #print (imagesList)
    f=open("/home/jay/beta/DCGAN/img_list.txt")
    image_list=[]
    for img_name in f:
        img_path = path+img_name[:-1]
        if img_path[-3:] == 'png':
            img = imread(img_path)
            if img.shape != (100,100,3):
                img=resize(img,(100,100,3))
            
            image_list.append(rgb2gray(img[3:97,3:97,:]))
            
    images_array=np.reshape(image_list,(len(image_list),94,94,1))
    return images_array
    #return image_list
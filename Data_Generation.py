# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 02:07:01 2019
##This script was used to generate data that was to feeded to my designed CNN(the archtecture in Notebook)
@author: MMOHTASHIM
"""
import cv2
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import time


os.chdir(r"C:\Users\MMOHTASHIM\Anaconda3\libs\Sleep-Deprivation-Project\appa-real-release")
def generate_data():
    """"
    

    The main purpose of this function was simple, it read the labels of images from csv files and then by using cv2 
    it read the desired image associated with the given label. 
    
    Returns=train_set-an array of shape(numberofimages,224,224,3)
    """"
    os.chdir(r"C:\Users\MMOHTASHIM\Anaconda3\libs\Sleep-Deprivation-Project\appa-real-release")
    df_train=pd.read_csv("gt_avg_train.csv")
    file_names=df_train["file_name"]##labels
    os.chdir(r"C:\Users\MMOHTASHIM\Anaconda3\libs\Sleep-Deprivation-Project\appa-real-release\train")
    train_set=[]##Main Data Array

    for i in tqdm(file_names):
        if i=='005613.jpg':##this if conditon was used so that I can read another csv file in the same loop and increase my training data
            os.chdir(r"C:\Users\MMOHTASHIM\Anaconda3\libs\Sleep-Deprivation-Project\appa-real-release\test")
        img=cv2.imread(i+"_face"+".jpg",1)
        resized_image = cv2.resize(img, (224, 224))##resize the image
        train_set.append(resized_image)##appending the data
    train_set=np.array(train_set)
    print(train_set.shape)
    os.chdir(r"C:\Users\MMOHTASHIM\Anaconda3\libs\Sleep-Deprivation-Project\appa-real-release")
    np.save("train_data.npy",train_set)
    
def generate_X_y():
    """"
    Loaded the saved training data into a variable X and then create labels(y).
    I read the real-age of the associated image from the csv file and then made the ages to be divided into four      
    categories=kid,youth,adult,elder through numerical threshold,this allowed my neural network work more efficently.
    
    Returns X,y(Final data)
    """"
    os.chdir(r"C:\Users\MMOHTASHIM\Anaconda3\libs\Sleep-Deprivation-Project\appa-real-release")
    X=np.load("train_data.npy")
    y=[]
    df_train=pd.read_csv("gt_avg_train.csv")
    age_data=df_train["real_age"]
    unique_labels=list(set(age_data))###Unique Ages
    print(unique_labels)
    for i in tqdm(age_data):##Going through all the real ages of images
        one_hot_label=np.zeros(4)##One hot age array
        if 1<=i<=11:#kid
            one_hot_label[0]=1##one-hot encoding
        elif 12<=i<=24:#youth
            one_hot_label[1]=1
        elif 24<=i<=60:#Adult
            one_hot_label[2]=1
        else:#elder
            one_hot_label[3]=1
        y.append(one_hot_label)
    y=np.array(y)
    np.save("train_data_label.npy",y)
    return X,y

if '__main__' == __name__:
    generate_data()
    
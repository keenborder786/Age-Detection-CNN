# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 02:07:01 2019

@author: MMOHTASHIM
"""
import cv2
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import time


os.chdir(r"C:\Users\MMOHTASHIM\Anaconda3\libs\Sleep-Deprivation-Project\appa-real-release")
def image_read():
    os.chdir(r"C:\Users\MMOHTASHIM\Anaconda3\libs\Sleep-Deprivation-Project\appa-real-release")
    df_train=pd.read_csv("gt_avg_train.csv")
    file_names=df_train["file_name"]
    os.chdir(r"C:\Users\MMOHTASHIM\Anaconda3\libs\Sleep-Deprivation-Project\appa-real-release\train")
    train_set=[]

    for i in tqdm(file_names):
        if i=='005613.jpg':
            os.chdir(r"C:\Users\MMOHTASHIM\Anaconda3\libs\Sleep-Deprivation-Project\appa-real-release\test")
        img=cv2.imread(i+"_face"+".jpg",1)
        resized_image = cv2.resize(img, (224, 224))
        train_set.append(resized_image)
    train_set=np.array(train_set)
    print(train_set.shape)
    os.chdir(r"C:\Users\MMOHTASHIM\Anaconda3\libs\Sleep-Deprivation-Project\appa-real-release")
    np.save("train_data.npy",train_set)
    
def generate_data():
    os.chdir(r"C:\Users\MMOHTASHIM\Anaconda3\libs\Sleep-Deprivation-Project\appa-real-release")
    X=np.load("train_data.npy")
    y=[]
    df_train=pd.read_csv("gt_avg_train.csv")
    age_data=df_train["real_age"]
    unique_labels=list(set(age_data))
    print(unique_labels)
    for i in tqdm(age_data):
        one_hot_label=np.zeros(4)
        if 1<=i<=11:
            one_hot_label[0]=1
        elif 12<=i<=24:
            one_hot_label[1]=1
        elif 24<=i<=60:
            one_hot_label[2]=1
        else:
            one_hot_label[3]=1
        y.append(one_hot_label)
    y=np.array(y)
    np.save("train_data_label.npy",y)
    return X,y

os.chdir(r"C:\Users\MMOHTASHIM\Anaconda3\libs\Sleep-Deprivation-Project")
root_logdir = os.path.join(os.curdir, "my_logs")
def get_run_logdir():
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)

if '__main__' == __name__:
    generate_data()
    
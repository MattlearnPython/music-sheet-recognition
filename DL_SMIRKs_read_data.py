#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  3 16:10:50 2018

@author: jinzhao
"""

import numpy as np
import os
import cv2   
from SMIRKs_head import *
    
def read_data(folder_name):
    
    raw_data = {}     
    for i in range(len(folder_name)):
        file_name = [] 
        for file in os.listdir(r'./%s' %(folder_name[i])):
            file_name.append(file) 
            
        file_name.remove('.DS_Store')
        
        img_set = []
        for j in range(len(file_name)):
            img = cv2.cvtColor(cv2.imread(r'./%s/%s' %(folder_name[i], file_name[j])), cv2.COLOR_BGR2GRAY)
            thresh, img_bw = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            img_bw = convert(img_bw) 
            img_set.append(img_bw)   
            
        raw_data['%s' %(folder_name[i])] = img_set
    return raw_data

# 'raw_data' and 'label_raw_data' are both dictionary.
def generate_data_set(raw_data, label_raw_data):
    
    train_X = np.ones([50*80, 1])
    train_y = np.ones([1, 1])
    for key in raw_data:
        note_set = raw_data[key] 
        for j in range(len(note_set)):
            img = note_set[j] / 255
            img_rz = cv2.resize(img, (50, 80), interpolation = cv2.INTER_NEAREST)
            img_ = img_rz.reshape(50*80, 1)
            
            train_X = np.column_stack((train_X, img_))
            label = label_raw_data[key]
            train_y = np.column_stack((train_y, label))
            
    train_X = np.delete(train_X, 0, axis = 1)
    train_y = np.delete(train_y, 0, axis = 1)
    return train_X, train_y
   




# =============================================================================
# half_note = data_set['half_note']
# quarter_note = data_set['quarter_note']
# eighth_note = data_set['eighth_note']  
# =============================================================================


# =============================================================================
# img = cv2.imread('music.png', cv2.COLOR_BGR2GRAY)
# img = cv2.resize(img, (25, 75))
# cv2.waitKey(1)
# cv2.destroyAllWindows() 
# =============================================================================
        
# =============================================================================
# img = raw_data['eighth_note'][0]
# img = convert(img)
# cv2.imshow('test', img)
# cv2.waitKey(1)
# cv2.destroyAllWindows() 
# =============================================================================

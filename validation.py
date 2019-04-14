# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 09:49:57 2019

@author: Yiqian Zhang
"""

from keras.preprocessing import image
import numpy as np
from keras import models
import time
import os
import tensorflow as tf 
import keras.backend.tensorflow_backend as KTF 
import keras_resnet
# CPU testing
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

# GPU configeration
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
session = tf.Session(config=config)
KTF.set_session(session)

root_path = '../dataset/driver_dataset/val/'
val_file = open('../dataset/driver_dataset/test_list.txt', 'r')
val_list = val_file.readlines()
#model = models.load_model('model/weights.best.resnet18.hdf5', custom_objects={'BatchNormalization': keras_resnet.layers.BatchNormalization, 'ResNet2D18':keras_resnet.models._2d.ResNet2D18})
model = models.load_model('model/weights.best.4layers.hdf5')

def topX_acc(preds, label):
    top_values = (-preds).argsort()[:3]
    if top_values[0] == label:
        top1 = True
    else:
        top1 = False
    if np.isin(label, top_values):
        top3 = True
    else:
        top3 = False
    return top1, top3

tic = time.time()
results = []
for _, l in enumerate(val_list):
    label, img_path = l.split()
    label = int(label)
    img_path = root_path + img_path
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255
    preds = model.predict(x)
    top1, top3 = topX_acc(preds[0], label)
    results.append([label, top1, top3])
toc = time.time()
print("total duration=", toc-tic, "each frame=", (toc-tic)/len(val_list))  

def results_report(results):
    top1_count = np.zeros(10)
    top3_count = np.zeros(10)
    for item in results:
        if item[1] == True:
            top1_count[item[0]] += 1
        if item[2] == True:
            top3_count[item[0]] += 1
    
    for index in range(10):
        print('-'*10 +'label '+str(index)+' results')
        print('top1='+str(top1_count[index]))
        print('top3='+str(top3_count[index]))
    
    print('-'*20)
    print('total top1='+str(np.sum(top1_count)))
    print('total top3='+str(np.sum(top3_count)))

results_report(results)


    






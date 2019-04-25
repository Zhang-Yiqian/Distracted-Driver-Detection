# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 09:49:57 2019

@author: Yiqian Zhang
"""

import tensorflow as tf

resnet_rec = dict(val_loss=None, val_acc=None, loss=None, acc=None)

# e，即event，代表某一个batch的日志记录
for e in tf.train.summary_iterator('./log_ResNet18/events.out.tfevents.1554031119.star-MS-7A93'):
    val_loss = []
    val_acc = []
    loss = []
    acc = []
    # v，即value，代表这个batch的某个已记录的观测值，loss或者accuracy
    for v in e.summary.value:
        if v.tag == 'loss':
            loss.append(v.simple_value)
        elif v.tag == 'accuracy':
            loss.append(v.simple_value)
        elif v.tag == 'val_loss':
            loss.append(v.simple_value)
        elif v.tag == 'acc':
            loss.append(v.simple_value)
    print(val_acc)






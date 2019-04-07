# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 18:25:04 2019

@author: Zhang_Yiqian
"""

import os
import random

ratio_train2all = 0.9

root = r"G:/essay/imgs"
train_list_file = open(root+"train_list.txt", "w")
test_list_file = open(root+"test_list.txt", "w")

first_directories = os.listdir(root)
train_list = []
test_list = []
for first_directory in first_directories:
    second_directories = os.listdir(os.path.join(root, first_directory))
    random.shuffle(second_directories)
    # format: label  path.  e.g. "1   c1/0001.jpg"
    second_directories = list(map(lambda second_directory:first_directory[-1]+"\t"+
                                  first_directory+'/'+second_directory+"\n", second_directories))
    train_set = second_directories[0:int(len(second_directories)*ratio_train2all)]
    test_set = second_directories[int(len(second_directories)*ratio_train2all):]
    train_list.extend(train_set)
    test_list.extend(test_set)
    print('lenth of train set'+first_directory[-1]+' is '+str(len(train_set)))
    print('lenth of test set'+first_directory[-1]+' is '+str(len(test_set)))

# make sure the train list is shuffled
random.shuffle(train_list)
train_list_file.writelines(train_list)
test_list_file.writelines(test_list)
train_list_file.close()
test_list_file.close()

#==============================================================================
# lenth of train set0 is 2240
# lenth of test set0 is 249
# lenth of train set1 is 2040
# lenth of test set1 is 227
# lenth of train set2 is 2085
# lenth of test set2 is 232
# lenth of train set3 is 2111
# lenth of test set3 is 235
# lenth of train set4 is 2093
# lenth of test set4 is 233
# lenth of train set5 is 2080
# lenth of test set5 is 232
# lenth of train set6 is 2092
# lenth of test set6 is 233
# lenth of train set7 is 1801
# lenth of test set7 is 201
# lenth of train set8 is 1719
# lenth of test set8 is 192
# lenth of train set9 is 1916
# lenth of test set9 is 213
#==============================================================================





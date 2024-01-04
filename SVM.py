#import libraries
import glob
import cv2
from PIL import Image
from natsort import natsorted
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow

#train, test, and val datasets
train = pd.read_csv("training_dataset/ground_truth/train.csv")
test = pd.read_csv("testing_dataset/Bone age ground truth.csv")
val = pd.read_csv('validation_dataset/boneage-validation-dataset.csv')


train['id'] = train['id'].astype(str)
test['Case ID'] = test['Case ID'].astype(str)
val['id'] = val['id'].astype(str)


train['id'] = train['id']+'.png'
test['Case ID'] = test['Case ID']+'.png'
val['id'] = val['id'] + '.png'

train['male'].replace({False : 0, True : 1}, inplace=True)
test['Sex'].replace({'F' : 0, 'M' : 1}, inplace=True)
val['male'].replace({False : 0, True : 1}, inplace=True)

print(train)
print(test)
print(val)

train_images = []
train_target = []
train_gender = []

test_images = []
test_target = []
test_gender = []

val_images = []
val_target = []
val_gender = []

import random
randomArray = []
for i in range(12611):
    randomArray.append(i)
list = random.sample(randomArray, 12611);
list.sort()

#load images and scale rgb values to be between 0 and 1
for i in range(0, len(list)):
#for i in range(0, 6000):
    img = tensorflow.keras.utils.load_img('training_dataset/boneage-training-dataset/'+train['id'][list[i]], color_mode="rgb", target_size=(128,128))
    #img = tensorflow.keras.utils.load_img('training_dataset/boneage-training-dataset/' + train['id'][i],
                                          #color_mode="grayscale", target_size=(128, 128))
    train_images.append(np.asarray(img) /255.)
    train_target.append(train['boneage'][i])
    train_images[i] = train_images[i].flatten()
    train_gender.append(train['male'][i])
    if(i % 1250 == 0):
        print(i, 'Image Train Loaded')
print('Train Image Loaded !!')

for i in range(0, len(test)):
    img = tensorflow.keras.utils.load_img('testing_dataset/boneage-test-dataset/boneage-test-dataset/'+test['Case ID'][i], color_mode='rgb', target_size=(128,128))
    test_images.append(np.asarray(img) /255.)
    test_target.append(test['Ground truth bone age (months)'][i])
    test_images[i] = test_images[i].flatten()
    test_gender.append(test['Sex'][i])

print('\nTest Image Loaded !!')

for i in range(0, len(val)):
    img = tensorflow.keras.utils.load_img('validation_dataset/boneage-validation/'+val['id'][i], color_mode='rgb', target_size=(128,128))
    val_images.append(np.asarray(img) /255.)
    val_target.append(val['boneage'][i])
    val_gender.append(val['male'][i])
print('\nValidation Image Loaded !!')
train_images = np.asarray(train_images)
train_target = np.asarray(train_target)
train_gender = np.asarray(train_gender)

test_images = np.asarray(test_images)
test_target = np.asarray(test_target)
test_gender = np.asarray(test_gender)

val_images = np.asarray(val_images)
val_target = np.asarray(val_target)
val_gender = np.asarray(val_gender)

train_gender.shape
train_gender = train_gender[:, np.newaxis]
test_gender = test_gender[:, np.newaxis]
test_gender.shape


train_data = np.concatenate((train_images, train_gender), axis=1)
train_data.shape

from sklearn import svm
#h1 = svm.SVR(C= 1, gamma= 'auto', kernel= 'rbf', max_iter= 30000)
#model=svm.SVR(C= 20, epsilon= 0.008, gamma=0.0003, max_iter=30000)
model=svm.SVR(kernel='rbf',max_iter=30000)
#h1 = svm.SVR(max_iter=30000)
model.fit(train_data, train_target)


test_data = np.concatenate((test_images, test_gender), axis=1)
#make predictions
test_data.shape
y_pred =  np.array(model.predict(test_data))
print(y_pred)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
print(mean_absolute_error(test_target, y_pred))
print(mean_absolute_percentage_error(test_target, y_pred))
print(mean_squared_error(test_target, y_pred))
print(r2_score(test_target, y_pred))
#train predictions
y_pred_train = np.array(model.predict(train_data))
#training accuracies
print(mean_absolute_error(train_target, y_pred_train))
print(mean_squared_error(train_target, y_pred_train))
print(r2_score(train_target, y_pred_train))

#mse
from statsmodels.tools.eval_measures import mse
print(mse(Y_train_mod, y_pred_train)) #training mse
print(mse(Y_test, y_pred))       #testing mse
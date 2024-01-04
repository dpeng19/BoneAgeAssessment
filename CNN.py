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


train = pd.read_csv("training_dataset/ground_truth/train.csv")
test = pd.read_csv("testing_dataset/Bone age ground truth.csv")
#val = pd.read_csv('validation_dataset/boneage-validation-dataset.csv')
val = pd.read_csv('validation_dataset/boneage-validation-dataset.csv')
val.sort_values(by = 'id', inplace=True)
val.reset_index(drop=True, inplace=True)
print(val)
print(train)

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

import random
randomArray2 = []
for i in range(1425):
    randomArray2.append(i)
list2 = random.sample(randomArray2, 1425);
list2.sort()

list2

#image = cv2.imread('training_dataset/boneage-training-dataset/1377.png', cv2.IMREAD_COLOR)
#image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
#final_img = cv2.resize(image_bw, dsize=(128, 128))

#clahe = cv2.createCLAHE(clipLimit=5)
#final_img[:, :, 0] = clahe.apply(final_img[:,:, 0])
#final_img = cv2.cvtColor(final_img, cv2.COLOR_LAB2RGB)
#cv2.imshow("image", final_img)
#plt.imshow(image, 'rgb')
#plt.show()
for i in range(0, len(list)):
#for i in range(0, 6000):
    image = cv2.imread('training_dataset/boneage-training-dataset/'+train['id'][list[i]],  cv2.IMREAD_COLOR)

    image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    final_img = cv2.resize(image_bw, dsize=(128, 128))

    clahe = cv2.createCLAHE(clipLimit=5)
    final_img[:, :, 0] = clahe.apply(final_img[:,:, 0])
    final_img = cv2.cvtColor(final_img, cv2.COLOR_LAB2RGB)
    #img = tensorflow.keras.utils.load_img('training_dataset/boneage-training-dataset/'+train['id'][list[i]], color_mode="rgb", target_size=(128,128))
    #img = tensorflow.keras.utils.load_img('training_dataset/boneage-training-dataset/' + train['id'][i],
                                          #color_mode="grayscale", target_size=(128, 128))
    #train_images.append(np.asarray(img) /255.)
    train_images.append(np.asarray(final_img) /255.)
    train_target.append(train['boneage'][i])
    train_gender.append(train['male'][i])
    if(i % 1250 == 0):
        print(i, 'Image Train Loaded')
print('Train Image Loaded !!')

for i in range(0, len(test)):
    image = cv2.imread('testing_dataset/boneage-test-dataset/boneage-test-dataset/'+test['Case ID'][i],  cv2.IMREAD_COLOR)

    image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    final_img = cv2.resize(image_bw, dsize=(128, 128))

    clahe = cv2.createCLAHE(clipLimit=5)
    final_img[:, :, 0] = clahe.apply(final_img[:, :, 0])
    final_img = cv2.cvtColor(final_img, cv2.COLOR_LAB2RGB)
    #img = tensorflow.keras.utils.load_img('testing_dataset/boneage-test-dataset/boneage-test-dataset/'+test['Case ID'][i], color_mode='rgb', target_size=(128,128))
    #test_images.append(np.asarray(img) /255.)
    test_images.append(np.asarray(final_img) / 255.)
    test_target.append(test['Ground truth bone age (months)'][i])
    test_gender.append(test['Sex'][i])

print('\nTest Image Loaded !!')

for i in range(0, len(list2)):
    image = cv2.imread('validation_dataset/boneage-validation/'+val['id'][list2[i]],  cv2.IMREAD_COLOR)

    image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    final_img = cv2.resize(image_bw, dsize=(128, 128))

    clahe = cv2.createCLAHE(clipLimit=5)
    final_img[:, :, 0] = clahe.apply(final_img[:, :, 0])
    final_img = cv2.cvtColor(final_img, cv2.COLOR_LAB2RGB)
    #img = tensorflow.keras.utils.load_img('validation_dataset/boneage-validation/'+val['id'][list2[i]], color_mode='rgb', target_size=(128,128))
    #val_images.append(np.asarray(img) /255.)
    val_images.append(np.asarray(final_img) / 255.)
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

train_images

from keras.layers import BatchNormalization, concatenate
from keras.layers import Dropout
from keras import optimizers, Input, Model
import keras
from keras import Model


inputA = Input(shape=(128,128,3))     #image input
inputB = Input(shape=(1,))            #gender input
x= keras.layers.Conv2D(16, kernel_size=(3, 3), activation='relu', padding = 'same')(inputA)
x = keras.layers.MaxPooling2D(pool_size=(2,2))(x)
x= keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding = 'same')(x)
x =keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

x= keras.layers.Conv2D(16, kernel_size=(3, 3), activation='relu', padding = 'same')(x)
x = keras.layers.MaxPooling2D(pool_size=(2,2))(x)
x =keras.layers.Flatten()(x)
#x =keras.layers.Dense(128, activation='relu')(x)
x =keras.layers.Dense(256, activation='relu')(x)
x =keras.layers.Dense(64, activation='relu')(x)
x =keras.layers.Dense(32, activation='relu')(x)
#x =keras.layers.Dense(16, activation='relu')(x)
x = Model(inputs=inputA, outputs=x)
# the second branch opreates on the second input
y = keras.layers.Dense(64, activation="relu")(inputB)
y = keras.layers.Dense(32, activation="relu")(y)
y = Model(inputs=inputB, outputs=y)
# combine the output of the two branches
combined = concatenate([x.output, y.output])
# apply a FC layer and then a regression prediction on the
# combined outputs
z = keras.layers.Dense(32, activation="relu")(combined)
z = keras.layers.Dense(16, activation="relu")(z)
#z = keras.layers.Dense(4, activation="relu")(z)
final = keras.layers.Dense(1, activation="linear")(z)
#-----delete----
#final = keras.layers.Dense(1, activation="linear")(x)
# our model will accept the inputs of the two branches and
# then output a single value
model = Model(inputs=[x.input, y.input], outputs=final)
#----delete-----
#model = Model(inputs=inputA, outputs=final)
model.compile(optimizers.Adam(learning_rate=0.0005), loss = 'mae')

model.save_weights('model.h5')

import keras.callbacks
history = model.fit(
    x=[train_images, train_gender],
    #x = [train_images],
    y=train_target,
    #validation_split=0.2,
    validation_data=([val_images, val_gender], val_target),
    #validation_data=([val_images], val_target),

    epochs=50,
    validation_batch_size=32,

    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
    ]
)

from sklearn.model_selection import GridSearchCV
param_grid = {'learning_rate': [1, 0.1, 1e-2, 1e-3, 1e-4, 1e-5, 0.0007]}
grid = GridSearchCV(model, param_grid, refit=True, verbose=3, n_jobs=-1, cv=3, scoring='neg_mean_absolute_error')

# fitting the model for grid search
grid.fit(combined, train_target)
train_images.shape
train_gender.shape

# print best parameter after tuning
print(grid.best_params_)
predicted_ages = np.squeeze(model.predict([test_images, test_gender]))
predicted_ages

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
print(mean_absolute_error(test_target, predicted_ages))
print(mean_squared_error(test_target, predicted_ages))
print(r2_score(test_target, predicted_ages))
print(mean_absolute_percentage_error(test_target, predicted_ages))

predicted_ages_train = np.squeeze(model.predict([train_images, train_gender]))
print(mean_absolute_error(train_target, predicted_ages_train))
print(mean_squared_error(train_target, predicted_ages_train))
print(r2_score(train_target, predicted_ages_train))


final_infer = pd.DataFrame()
final_infer['Predicted'] = predicted_ages
final_infer['Orginal'] = test_target
final_infer
final_infer['Predicted'] = (predicted_ages/12.).round()
final_infer['Orginal'] = (test_target/12.).round()
for i in range(200):
    print(str(final_infer['Predicted'][i]) + " " + str(final_infer['Orginal'][i])+ "\n")

print(mean_absolute_error(final_infer['Orginal'], final_infer['Predicted']))

newdf = final_infer[final_infer['Orginal'] <= 120]
print(mean_squared_error(newdf['Orginal'], newdf['Predicted']))
print(mean_absolute_error(newdf['Orginal'], newdf['Predicted']))



model.load_weights('model.h5')




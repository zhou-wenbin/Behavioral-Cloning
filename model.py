import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Cropping2D
from keras.layers import Activation, Convolution2D
from keras.layers.normalization import BatchNormalization
import csv
import argparse
import os
import cv2
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation,Lambda
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Input, ELU
from keras.models import load_model, model_from_json
from sklearn.utils import shuffle
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
import matplotlib as plt


def resize_img(image):
    shape = image.shape
    #Cut off the sky from the original picture
    crop_up = shape[0]-100
    #Cut off the front of the car
    crop_down = shape[0]-25

    image = image[crop_up:crop_down, 0:shape[1]]
    #resize dimmension to 64x64 for the specific NN by NVIDIA 
    dim= (64,64)
    image = cv2.resize(image,dim, interpolation = cv2.INTER_AREA)
    return image

# def rgb2yuv(image):
#     """
#     Convert the image from RGB to YUV (This is what the NVIDIA model does)
#     """
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

#     return image

def image_process(line, data_path):
    thresh = (40, 255)

    steering_center = float(line[3])
    # create adjusted steering measurements for the side camera images
    correction = 0.22# this is a parameter to tune
    steering_left = steering_center + correction
    steering_right = steering_center - correction
    
    source_path_center = line[0]
    source_path_left = line[1]
    source_path_right = line[2]
    
    filename_center = source_path_center.split('/')[-1]
    filename_left  =  source_path_left.split('/')[-1]
    filename_right  =  source_path_right.split('/')[-1]
    
    current_path_center = data_path+ filename_center
    current_path_left = data_path+ filename_left
    current_path_right = data_path + filename_right
    
    
    image_center = cv2.imread(current_path_center)

    # generate a flipped image
    augmented_images2= cv2.flip(image_center,1)
    augmented_images2 = resize_img(augmented_images2)
    
    # use s channel of a HLS image to process the image. after processing
    # we are left with only road information and other information.
    hls = cv2.cvtColor(augmented_images2, cv2.COLOR_RGB2HLS)
    S = hls[:,:,2]
    augmented_images2 = np.zeros_like(S)
    augmented_images2[(S > thresh[0]) & (S <= thresh[1])] = 1
    arr1 = np.zeros(64*64)
    arr1= arr1.reshape((64,64))
    arr2 =  np.zeros(64*64)
    arr2= arr2.reshape((64,64))
    augmented_images2=augmented_images2*255
    augmented_images2=np.stack((augmented_images2,arr1,arr2),axis=2)


    augmented_angle = steering_center*-1.0
    
    # transfer center image to YUV channel, crop and resize
    # image_center2 = rgb2yuv(image_center)
    image_center2 = resize_img(image_center)
    hls = cv2.cvtColor(image_center2, cv2.COLOR_RGB2HLS)
    S = hls[:,:,2]
    image_center2 = np.zeros_like(S)
    image_center2[(S > thresh[0]) & (S <= thresh[1])] = 1
    arr1 = np.zeros(64*64)
    arr1= arr1.reshape((64,64))
    arr2 =  np.zeros(64*64)
    arr2= arr2.reshape((64,64))
    image_center2=image_center2*255
    image_center2=np.stack((image_center2,arr1,arr2),axis=2)



    
    # transfer left image to YUV channel, crop and resize
    image_left = cv2.imread(current_path_left)
    # image_left2 = rgb2yuv(image_left)
    image_left2 = resize_img(image_left)
    hls = cv2.cvtColor(image_left2, cv2.COLOR_RGB2HLS)
    S = hls[:,:,2]
    image_left2 = np.zeros_like(S)
    image_left2[(S > thresh[0]) & (S <= thresh[1])] = 1
    arr1 = np.zeros(64*64)
    arr1= arr1.reshape((64,64))
    arr2 =  np.zeros(64*64)
    arr2= arr2.reshape((64,64))
    image_left2=image_left2*255
    image_left2=np.stack((image_left2,arr1,arr2),axis=2)


    # transfer right image to YUV channel, crop and resize
    image_right = cv2.imread(current_path_right)
    # image_right2 = rgb2yuv(image_right)
    image_right2 = resize_img(image_right)

    hls = cv2.cvtColor(image_right2, cv2.COLOR_RGB2HLS)
    S = hls[:,:,2]
    image_right2 = np.zeros_like(S)
    image_right2[(S > thresh[0]) & (S <= thresh[1])] = 1
    arr1 = np.zeros(64*64)
    arr1= arr1.reshape((64,64))
    arr2 =  np.zeros(64*64)
    arr2= arr2.reshape((64,64))
    image_right2=image_right2*255
    image_right2=np.stack((image_right2,arr1,arr2),axis=2)
    
    # save the processed image data with balance
    car_images.append(image_center2)
    car_images.append(image_left2)
    car_images.append(cv2.flip(image_left2,1))
    car_images.append(image_right2)
    car_images.append(cv2.flip(image_right2,1))
    car_images.append(augmented_images2)
    
    #save corresponding steering angles.
    steering_angles.append(steering_center)
    steering_angles.append(steering_left)
    steering_angles.append(steering_left*-1.0)
    steering_angles.append(steering_right)
    steering_angles.append(steering_right*-1.0)
    steering_angles.append(augmented_angle)




def collect_data(path):
    """
    filter the data wih a random ratio 1/4 on data with steering angle 0
    """
    
    lines=[]
    drive_log_path = path + "driving_log.csv"
    data_path = path + "IMG/"
    with open(drive_log_path) as csvfile:
        reader=csv.reader(csvfile)
        next(csvfile)
        for line in reader:
            lines.append(line)
    for line in lines:
        if float(line[3]) != 0:
            image_process(line,data_path)       
        else:
            prob = np.random.uniform()
            if prob <= 0.25:
                image_process(line,data_path)
                

    

path = "/home/workspace/data/"
# path1= "/home/workspace/big_turn/"
path2 = "/home/workspace/big_turn_good/"
path3= "/home/workspace/right_turn/"
# path4 = "/home/workspace/zigzag_data/"
path5 = "/home/workspace/tekitou_data/"
path6 = "/home/workspace/more_data/"



car_images= []
steering_angles = []

collect_data(path);
# collect_data2(path1);
# collect_data(path2);
# collect_data(path1);
# collect_data(path2);
# collect_data2(path3);
# collect_data(path4);
# collect_data(path5);
# collect_data(path6);


print("data collected....")





Xtrain = np.array(car_images)
ytrain = np.array(steering_angles)


from sklearn.utils import shuffle

Xtrain , ytrain = shuffle(Xtrain , ytrain, random_state=0)

from sklearn.model_selection import train_test_split

# Split the data
X_train, X_val, y_train, y_val = train_test_split(Xtrain, ytrain, test_size=0.10, shuffle= True)

from keras.preprocessing.image import ImageDataGenerator


datagen = ImageDataGenerator()
val_datagen = ImageDataGenerator()


batch_size = 64
epochs = 9
# Note: we aren't using callbacks here since we only are using 5 epochs to conserve GPU time



import time

print("start to train ....")

time.sleep(2.4)

model = Sequential()
model.add(Lambda(lambda x: (x/127.5) -1.0, input_shape=(64, 64, 3)))
model.add(Convolution2D(32, 3, 3, border_mode='same', subsample=(2, 2), activation='relu', name='Conv1'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='same'))
model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(2, 2), activation='relu', name='Conv2'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='same'))
model.add(Convolution2D(128, 3, 3, border_mode='same', subsample=(1, 1), activation='relu', name='Conv3'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='same'))
model.add(Convolution2D(128, 2, 2, border_mode='same', subsample=(1, 1), activation='relu', name='Conv4'))

model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu', name='FC1'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu', name='FC2'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu', name='FC3'))
model.add(Dense(1))



adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='mse',optimizer='adam')




model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size), 
                    steps_per_epoch=len(X_train)/batch_size, epochs=epochs, verbose=1, 
                    validation_data=val_datagen.flow(X_val, y_val, batch_size=batch_size),
                    validation_steps=len(X_val)/batch_size)

model.save('model.h5')


print("finish training")

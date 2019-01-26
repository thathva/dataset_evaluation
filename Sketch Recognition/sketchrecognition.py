# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 14:53:58 2019

@author: harip
"""
#importing libraries
from keras import optimizers
import numpy as np
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense
from keras.models import Sequential

#layers
classifier=Sequential()

#first layer
classifier.add(Conv2D(32,(3,3),input_shape=(64,64,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#second layer
classifier.add(Conv2D(32,(3,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Conv2D(32,(3,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))


#flatten it
classifier.add(Flatten())

#output layer
classifier.add(Dense(units=128,activation='relu'))
classifier.add(Dense(units=51,activation='sigmoid'))
adam = optimizers.Adam(lr=0.001)
#compile
classifier.compile(optimizer=adam, loss='categorical_crossentropy',metrics=['accuracy'])
#import glob
#import os
#rasterList = glob.glob(os.path.join('D:/Hackathon/dataset/training', '*.jpg'))
##Splitting data into training and testing
#
#from sklearn.model_selection import train_test_split
#train_samples, validation_samples = train_test_split(rasterList, test_size=0.2)

from keras.preprocessing.image import ImageDataGenerator

#load data
training_data=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)

testing_data=ImageDataGenerator(rescale=1./255)

#fit data
training_set=training_data.flow_from_directory('F:/dataset/training',
                                               target_size=(64,64),
                                               batch_size=32,
                                               class_mode='categorical',
                                               )
testing_set=testing_data.flow_from_directory('F:/dataset/testing',
                                             target_size=(64,64),
                                             batch_size=32,
                                             class_mode='categorical',
                                             )
classifier.fit_generator(training_set,
                         steps_per_epoch=120,
                         epochs=30,
                         validation_data=testing_set,
                         validation_steps=991)
classifier.summary()

from keras.models import load_model

classifier.save('my_model.h5') 

#from keras.preprocessing import image
#test_image = image.load_img('C:/Users/harip/Desktop/ben.jpg', target_size = (64, 64))
#test_image = image.img_to_array(test_image)
#test_image = np.expand_dims(test_image, axis = 0)
#result = classifier.predict_classes(test_image)
#print(result)
training_set.class_indices
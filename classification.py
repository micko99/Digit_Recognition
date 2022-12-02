import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, AveragePooling2D
from keras.losses import CategoricalCrossentropy, SparseCategoricalCrossentropy
from keras.optimizers import Adam
from keras.utils import normalize, to_categorical
import os
import cv2
from shutil import rmtree

class Classification:
    '''
        CNN classification or NN classification define by parameter type
    '''
    
    def __init__(self, type):
        self.type = type

        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
        self.x_train = normalize(self.x_train, axis=1)
        self.x_test = normalize(self.x_test, axis=1)
        if self.type == 'cnn':
            self.x_train = np.array(self.x_train).reshape(-1, 28, 28, 1)
            self.x_test = np.array(self.x_test).reshape(-1, 28, 28, 1)

        self.model = None
        self.batch_images = []
    
    def prepare_model(self):
        if self.type == 'cnn':
            self.model = Sequential([
                    Input(shape=(28, 28, 1)),
                    Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
                    MaxPooling2D(pool_size=(2, 2)),
    
                    Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu'),
                    MaxPooling2D(pool_size=(2, 2)),
    
                    Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu'),
                    MaxPooling2D(pool_size=(2, 2)),
                    Dropout(0.25),

                    Flatten(),
                    Dense(64, activation='relu'),
                    Dense(10, activation='softmax')
                    ])
        else: 
            self.model = Sequential([
                Input(shape=(28, 28)),
                Flatten(),
                Dense(32, activation='relu'),
                Dense(32, activation='relu'),
                Dropout(0.25),
                Dense(32, activation='relu'),
                #Dropout(0.2),
                #Dense(16, activation='relu'),
                Dense(10, activation='softmax')
            ])

        self.model.compile(loss=SparseCategoricalCrossentropy(), optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    
    def train_model(self):
        self.prepare_model()
        if self.type == 'cnn':
            self.model.fit(self.x_train, self.y_train, epochs=6, validation_split=0.3)
            self.model.evaluate(self.x_test, self.y_test)
            self.model.save('digits_final.model')
        else:
            self.model.fit(self.x_train, self.y_train, epochs = 10, validation_split=0.2)
            self.model.evaluate(self.x_test, self.y_test)
            self.model.save('digits_final_nn.model')
    
    def processing_images(self):
        path = 'images'
        for filename in os.listdir(path):
            f = os.path.join(path, filename)
            if os.path.isfile(f):
                image = cv2.imread(f)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                new_image = 255 - gray
                new_image = cv2.resize(new_image, (28, 28), interpolation=cv2.INTER_AREA)
                new_image = normalize(new_image, axis=1)
                if self.type == 'cnn':
                    new_image = np.array(new_image).reshape(28, 28, 1) # (1, 28, 28, 1)
            
                self.batch_images.append(new_image)
    
    def predict_model(self):
        if self.type == 'cnn':
            self.model = load_model('digits_final.model')
        else:
            self.model = load_model('digits_final_nn.model')

        self.processing_images()
        prediction = self.model.predict(np.array(self.batch_images))
        prediction = np.argmax(prediction, axis=1)
        self.batch_images = []

        rmtree('images') # delete folder with roi images and create for using again
        os.mkdir('images')

        number = ''
        for x in prediction:
            number += str(x)
        return number


import numpy as np
import keras as ke
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Activation, Input, Flatten, Dense
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt


#Absolut paths have to get adjusted
train_path = '/Users/max/Documents/Proggerzeugs/PYTHONworkspace/ageEstimation/Age_Estimation/train'
valid_path = '/Users/max/Documents/Proggerzeugs/PYTHONworkspace/ageEstimation/Age_Estimation/valid'
test_path = '/Users/max/Documents/Proggerzeugs/PYTHONworkspace/ageEstimation/Age_Estimation/test'

#Batch sizes have to get adjusted
train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size=(224,224), classes=['man','woman'], batch_size=6)
valid_batches = ImageDataGenerator().flow_from_directory(valid_path, target_size=(224,224), classes=['man','woman'], batch_size=6)
test_batches = ImageDataGenerator().flow_from_directory(test_path, target_size=(224,224), classes=['man','woman'], batch_size=6)

#Plot images
def plots(ims, figsize=(12,6), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')
       
    imgs, labels = next(train_batches)

#Get Lables
plots(imgs, titles=labels)

vgg16_model = ke.applications.VGG16(weights='imagenet', include_top=False)



inputS = Input(shape=(224,224,3), name = 'image_input')

output_vgg16_model = vgg16_model(inputS)

x = Flatten(name='flatten')(output_vgg16_model)
x = Dense(4096, activation='relu', name='fc1')(x)
x = Dense(4096, activation='relu', name='fc2')(x)
x = Dense(2, activation='softmax', name='predictions')(x)

my_model = Model(inputs=inputS, outputs=x)

for layer in my_model.layers:
    layer.trainable = False

#Here we have to decide, how many layers we want to leave trainable:
#For the beginning we can test with 0 trainable layers, so comment the 
#following two lines out for the first test
for layer in my_model.layers[-1:]:
    layer.trainable = True
    
my_model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])

my_model.summary()

#30 images in the trian folder and a batch size of 6, leads to a step_per_epoch number of 5, 5*6=30
my_model.fit_generator(train_batches, steps_per_epoch=5, 
                   validation_data=valid_batches, validation_steps=5, epochs=5, verbose=2)
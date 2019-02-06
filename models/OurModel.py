import sys
sys.path.append("/IPNeuronaleNetze")
sys.path.append("/IPNeuronaleNetze/trainers")
import tensorflow as tf
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Flatten, Dense, GlobalAveragePooling2D, Input
from tensorflow.python.keras.optimizers import SGD



class OurModel:
    def __init__(self, identifier):
        self.model = self.build_model(identifier)

    def build_model(self, identifier):        
        # Load VGG16                    
        base_model = VGG16(weights='imagenet', include_top=False)
        fyipg = base_model.output

        # Add a global spatial average pooling layer
        fyipg = GlobalAveragePooling2D()(fyipg)
        # Let's add a fully-connected layer
        fyipg = Flatten(name ='Flatten1')(fyipg)
        fyipg = Dense(4096, activation='relu', name='AdditianlLayer1')(fyipg)
        fyipg = Dense(4096, activation='relu', name='AdditianlLayer2')(fyipg)
        if identifier == 1:
            fyipg = Dense(1, activation='sigmoid', name='Predictions')(fyipg)
        else:
            fyipg = Dense(101, activation='softmax', name='Predictions')(fyipg)

        # This is the model we will train       
        model = Model(inputs = base_model.input , outputs =  fyipg)
        # Setting optimizer for model                
        optimizer =tf.train.GradientDescentOptimizer(learning_rate = 0.0001)
        # Optimize model for gender- and agemodel
        if identifier == 1:
            model.compile(optimizer=optimizer,
                                loss='binary_crossentropy', metrics=['mae','acc'])
        else:
            model.compile(optimizer= optimizer,
                                loss='categorical_crossentropy', metrics=['mae', 'acc'])
        return model                                                                                                                                 
                                                                                                                              
    def load_model(self, filepath, identifier):
        # Load the save_model file
        self.model = tf.contrib.saved_model.load_keras_model(filepath)                                                        
        # Setting optimizer for model                
        optimizer =tf.train.GradientDescentOptimizer(learning_rate = 0.0001)                                                                                                                              
        # Optimize model for gender- and agemodel
        if identifier == 1:
            self.model.compile(optimizer=optimizer,                                                                                
                                loss='binary_crossentropy', metrics=['mae','acc'])
        else:
            self.model.compile(optimizer= optimizer,                                                                               
                                loss='categorical_crossentropy', metrics=['mae', 'acc'])                                                                                                                              
        return self.model
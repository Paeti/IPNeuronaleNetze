# import sys, os
# parent_dir = os.getcwd()
# sys.path.append(parent_dir)
import keras 
from tensorflow.keras.applications.vgg16 import VGG16
#from models.model import OurModel
#from trainers.trainer import Trainer
import tensorflow as tf
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras.backend import eval
#from LR_SGD import LR_SGD
import numpy as np
from keras.legacy import interfaces
from keras.optimizers import Optimizer
from keras.optimizers import adam

# class LR_SGD(Optimizer):
#     """Stochastic gradient descent optimizer.

#     Includes support for momentum,
#     learning rate decay, and Nesterov momentum.

#     # Arguments
#         lr: float >= 0. Learning rate.
#         momentum: float >= 0. Parameter updates momentum.
#         decay: float >= 0. Learning rate decay over each update.
#         nesterov: boolean. Whether to apply Nesterov momentum.
#     """

#     def __init__(self, lr=0.0001, momentum=0., decay=0.,
#                  nesterov=False,multipliers=None,**kwargs):
#         super(LR_SGD, self).__init__(**kwargs)
#         with K.name_scope(self.__class__.__name__):
#             self.iterations = K.variable(0, dtype='int64', name='iterations')
#             self.lr = K.variable(lr, name='lr')
#             self.momentum = K.variable(momentum, name='momentum')
#             self.decay = K.variable(decay, name='decay')
#         self.initial_decay = decay
#         self.nesterov = nesterov
#         self.lr_multipliers = multipliers

#     @interfaces.legacy_get_updates_support
#     def get_updates(self, loss, params):
#         grads = self.get_gradients(loss, params)
#         self.updates = [K.update_add(self.iterations, 1)]

#         lr = self.lr
#         if self.initial_decay > 0:
#             lr *= (1. / (1. + self.decay * K.cast(self.iterations,
#                                                   K.dtype(self.decay))))
#         # momentum
#         shapes = [K.int_shape(p) for p in params]
#         moments = [K.zeros(shape) for shape in shapes]
#         self.weights = [self.iterations] + moments
#         for p, g, m in zip(params, grads, moments):
            
#             matched_layer = [x for x in self.lr_multipliers.keys() if x in p.name]
#             if matched_layer:
#                 new_lr = lr * self.lr_multipliers[matched_layer[0]]
#             else:
#                 new_lr = lr

#             v = self.momentum * m - new_lr * g  # velocity
#             self.updates.append(K.update(m, v))

#             if self.nesterov:
#                 new_p = p + self.momentum * v - new_lr * g
#             else:
#                 new_p = p + v

#             # Apply constraints.
#             if getattr(p, 'constraint', None) is not None:
#                 new_p = p.constraint(new_p)

#             self.updates.append(K.update(p, new_p))
#         return self.updates

#     def get_config(self):
#         config = {'lr': float(K.get_value(self.lr)),
#                   'momentum': float(K.get_value(self.momentum)),
#                   'decay': float(K.get_value(self.decay)),
#                   'nesterov': self.nesterov}
#         base_config = super(LR_SGD, self).get_config()
#         return dict(list(base_config.items()) + list(config.items())) 


class OurModel:
    def __init__(self, identifier):
        self.model = self.buildModel(identifier)

    def buildModel(self, identifier): 

        input_layer = Input(shape=(224,224,3))
        newModel = VGG16(weights="imagenet", include_top=False)(input_layer) #input_shape = (224, 224, 3)
       
        # Define the input
        xInput =  newModel  
        # Add the fully-connected layers
        x = Flatten(name='flatten')(xInput)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x) 

        if identifier == 1:
            x = Dense(1, activation='sigmoid', name='predictions')(x)
        else:
            x = Dense(101, activation='softmax', name='predictions')(x)
        
        # Create our own model
        model = Model(inputs=input_layer, outputs=x)

        # Setting the Learning rate multipliers
        LR_mult_dict = {}
        LR_mult_dict['flatten'] = 100
        LR_mult_dict['fc1'] = 100
        LR_mult_dict['fc2'] = 100   
        LR_mult_dict['predictions'] = 100 

        # Setting optimizer for model
        #optimizer = LR_SGD(lr=0.0001, momentum=0.9, decay=0.0005, nesterov=True, multipliers = LR_mult_dict)
        #optimizer = SGD(lr=0.0001, decay=0.0005,
        #                           momentum=0.9, nesterov=True)  

        # Optimize VGG16 for gender- and agemodel
        if identifier == 1:
            model.compile(optimizer=adam(lr=0.001, decay=1e-6),
                                loss='binary_crossentropy')
        else:
            model.compile(optimizer=adam(lr=0.001, decay=1e-6),
                                loss='categorical_crossentropy')    
        return model


class build_modelTest(tf.test.TestCase):
    # def test_Age_differently_than_VGG16(self):
    #     with self.test_session():
    #         VGG16model = VGG16(weights="imagenet", include_top=False)
    #         AgeModel = OurModel(0)            
    #         self.assertNotEqual(VGG16model, AgeModel.model)
    
    # def test_Gender_differently_than_VGG16(self):
    #     with self.test_session():
    #         VGG16model = VGG16(weights="imagenet", include_top=False)
    #         GenderModel = OurModell(1)
    #         self.assertNotEqual(VGG16model, GenderModel.model)

    def test_Gender_layersLength(self):
        with self.test_session():
            GenderModel = OurModel(1)
            length=0
            for layer in GenderModel.model.layers:
                length = length+1       
            self.assertEqual(length, 20)
    
    def test_Age_layerLength(self):
        with self.test_session():
            AgeModel = OurModel(0)
            length=0
            for layer in AgeModel.model.layers:
                length = length+1       
            self.assertEqual(length, 20)

    def test_Model_layers_allTrainable(self):
        with self.test_session():
            GenderModel = OurModel(1)
            trainable = True
            for layer in GenderModel.model.layers:
                if layer.trainable != True:
                    trainable = False
            AgeModel = OurModel(0)
            for layer in AgeModel.model.layers:
                if layer.trainable != True:
                    trainable = False
            self.assertEqual(trainable, True)    

#TODO
# class train_modelTest(tf.test.TestCase):
#     def test_trainable_weights(self):
#         with self.test_session():
#             model = OurModell(1)  
#  
#             modelWeights = []
#             for layer in model.model.layers:
#                 modelWeights.append(layer.weights) 

#             trainer = Trainer(model, filepath, save_model_path....)
#             trainer.training()

#             modelWeightsAfterTraining = []
#             for layer in model.model.layers:
#                 modelWeightsAfterTraining.append(layer.weights)                 
#             self.assertAllNotEqual(modelWeights , modelWeightsAfterTraining) 
            
#        def test_weights_has_changed(tf.test.TestCase):
#            with self.test_session():
#                model = OurModell(1) 
#
#                  self.assertNotEqual(model, mode_after_train)

if __name__ == '__main__':
    tf.test.main()
  
   
    
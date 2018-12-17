import sys, os
parent_dir = os.getcwd()
sys.path.append(parent_dir)
import keras 
from tensorflow.keras.applications.vgg16 import VGG16
from models.model import OurModel
#from trainers.trainer import Trainer
import tensorflow as tf

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
            GenderModel = OurModell(1)
            length=0
            for layer in GenderModel.model.layers:
                length = length+1       
            self.assertEqual(length, 20)
    
    def test_Age_layerLength(self):
        with self.test_session():
            AgeModel = OurModell(0)
            length=0
            for layer in AgeModel.model.layers:
                length = length+1       
            self.assertEqual(length, 20)

    def test_Model_layers_allTrainable(self):
        with self.test_session():
            GenderModel = OurModell(1)
            trainable = True
            for layer in GenderModel.model.layers:
                if layer.trainable != True:
                    trainable = False
            AgeModel = OurModell(0)
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
  
   
    
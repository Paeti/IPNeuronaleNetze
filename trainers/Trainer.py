import sys, os
parent_dir = os.getcwd()
sys.path.append("/home/ip/IPNeuronaleNetze")
import tensorflow as tf
from models.dataloaders.DataLoader import DataLoader
from tensorflow import keras 
from tensorflow.python.keras import callbacks
from models.optimizer.cb import Cback
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.applications.vgg16 import preprocess_input
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt

class Trainer:
	
	def __init__(self, model, filepath, validationfilepath, testfilepath, identifier):		
		self.filepath = filepath
		self.model = model		
		self.saved_model_path = ""
		self.identifier = identifier
		self.validationfilepath = validationfilepath
		self.testfilepath = testfilepath
		#self.Trainer = self.training()
		self.Trainer = self.train()

	def train(self):
			
		batch_size = 12
		if self.identifier == 1:
			self.saved_model_path = "/home/ip/IPNeuronaleNetze/models/GenderWeights"
			weights_name = "model_gender.h5"
			results_name = "results_imdb_gender.csv"
			class_mode ='binary'
		else:
			self.saved_model_path = "/home/ip/IPNeuronaleNetze/models/AgeWeights"
			weights_name = "model_age.h5"
			results_name = "results_imdb_age.csv"
			class_mode = 'categorical'			
	
		datagen = ImageDataGenerator(preprocessing_function = preprocess_input)				
		
			
		train_generator = datagen.flow_from_directory(self.filepath, target_size = (224,224), batch_size=batch_size,
			class_mode = class_mode, shuffle = True)
                
		validation_generator = datagen.flow_from_directory(self.validationfilepath, target_size =(224,224),
			batch_size = batch_size, class_mode = class_mode, shuffle = True)
		
		test_generator = datagen.flow_from_directory(self.testfilepath, target_size =(224,224), batch_size = 1,
			class_mode = class_mode, shuffle = False)
			
				
		callbacks = Cback()
		callbacks = callbacks.makeCb()
		
		history = self.model.fit_generator(train_generator, steps_per_epoch = train_generator.samples/train_generator.batch_size ,validation_data=validation_generator, validation_steps = validation_generator.samples/validation_generator.batch_size, epochs = 312 ,callbacks = callbacks)
						
		self.model.save_weights(weights_name)

		self.saved_model_path = tf.contrib.saved_model.save_keras_model(
             self.model, self.saved_model_path, custom_objects=None, as_text=None)
		
		scores = self.model.evaluate_generator(test_generator,test_generator.samples)
		test_generator.reset()
		pred = self.model.predict_generator(test_generator, verbose=1)
		
		print("Evaluation Loss: ", scores[0], "Evaluation MAE: ", scores[1])
		if self.identifier == 1:
			predicted_class_indices = np.around(pred)
		else:
			predicted_class_indices = np.argmax(pred, axis=1)
		labels = (test_generator.class_indices)
		labels = dict((v,k) for k,v in labels.items())
		predictions = [labels[k] for k in predicted_class_indices[:,0]]
		filenames = test_generator.filenames
		results = pd.DataFrame ({"Filename":filenames,
					"Predictions":predictions})
		results.to_csv(results_name, index = False)				
	
		
		return self.model		

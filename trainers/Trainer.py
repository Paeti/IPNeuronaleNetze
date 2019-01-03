import sys, os
parent_dir = os.getcwd()
sys.path.append("/Users/ronnyaretz/IPNeuronaleNetze")

import tensorflow as tf
from models.dataloaders.DataLoader import DataLoader
from tensorflow import keras 
from tensorflow.python.keras import callbacks
from models.optimizer.cb import Cback
import json

class Trainer:
	
	def __init__(self, model, filepath, validationfilepath, identifier):		
		self.filepath = filepath
		self.model = model		
		self.saved_model_path = ""
		self.identifier = identifier
		self.validationfilepath = validationfilepath
		self.Trainer = self.training()

	def training(self):

		if self.identifier==1:
			self.saved_model_path = "IPNeuronaleNetze/models/GenderWeights"
		else:
			self.saved_model_path = "IPNeuronaleNetze/models/AgeWeights"

		dataLoader = DataLoader(self.filepath,self.identifier)		
		images, labels = dataLoader.create_dataset()

		validationDataLoader = DataLoader(self.validationfilepath, self.identifier)
		valdata = dataLoader.create_dataset()
		
		callbacks = Cback()
		callbacks = callbacks.makeCb()
		
		self.model.fit(x=images, y=labels, validation_data=valdata ,steps_per_epoch=25, epochs=125, callbacks=callbacks)			
		
		tf.contrib.saved_model.save_keras_model(
             self.model, self.saved_model_path, custom_objects=None, as_text=None)	
		return self.model
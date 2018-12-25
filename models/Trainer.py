import sys, os
parent_dir = os.getcwd()
sys.path.append(parent_dir)

import tensorflow as tf
from DataLoader import DataLoader
from tensorflow import keras 
from keras import callbacks
from optimizer.cb import Cback
import json

class Trainer:
	
	def __init__(self, model, filepath, identifier):		
		self.filepath = filepath
		self.model = model		
		self.saved_model_path = 'C:\\Users\\ckrem\\Desktop\\IP'
		self.identifier = identifier
		self.Trainer = self.training()    
		
	

	def training(self):
		dataLoader = DataLoader(self.filepath)		
		images, labels = dataLoader.create_dataset()		
		
		callbacks = Cback()
		callbacks = callbacks.makeCb()
		
		self.model.fit(x=images, y=labels, steps_per_epoch=5, epochs=1, callbacks= callbacks)			
		
		if self.identifier == 1:
			self.model.save_weights("GenderModel_weights.h5")
		else:
			self.model.save_weights("AgeModel_weights.h5")
		
		# tf.contrib.saved_model.save_keras_model(
        #     self.model, self.saved_model_path, custom_objects=None, as_text=None)	
		return self.model
		
	
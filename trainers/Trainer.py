import sys, os
parent_dir = os.getcwd()
sys.path.append("/home/ip/IPNeuronaleNetze")

import tensorflow as tf
from models.dataloaders.DataLoader import DataLoader
from tensorflow import keras 
from tensorflow.python.keras import callbacks
from models.optimizer.cb import Cback
import json
import matplotlib.pyplot as plt
import numpy as np

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
			self.saved_model_path = "/home/ip/IPNeuronaleNetze/models/GenderWeights"
		else:
			self.saved_model_path = "/home/ip/IPNeuronaleNetze/models/AgeWeights"

		dataLoader = DataLoader(self.filepath,self.identifier)		
		images, labels  = dataLoader.create_dataset()
		validationDataLoader = DataLoader(self.validationfilepath, self.identifier)
		images_val, labels_val = validationDataLoader.create_dataset()		
		
		callbacks = Cback()
		callbacks = callbacks.makeCb()
		
		self.model.fit(x= images, y = labels, validation_data=(images_val, labels_val), steps_per_epoch=100, validation_steps=35, epochs=10, callbacks=callbacks,shuffle = True)			
		
	#	tf.contrib.saved_model.save_keras_model(
        #     self.model, self.saved_model_path, custom_objects=None, as_text=None)	
		self.model.save_weights("model_shuffle.h5")
		return self.model

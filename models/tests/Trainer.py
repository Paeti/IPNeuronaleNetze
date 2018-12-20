import tensorflow as tf

from tensorflow import keras 
from DataLoader import DataLoader
from keras import callbacks
from cb import Cback
import json

class Trainer:
	
	def __init__(self, model, filepath):
		self.batchsize =  100
		self.epochs =  50
		self.filepath = filepath
		self.model = model		
		#self.saved_model_path = saved_model_path		
		#self.validation_dataset_filepath =  validation_dataset_filepath
		self.steps_per_epoch = 10000/self.batchsize #Noch anpassen (SUM_OF_ALL_DATASAMPLES / BATCHSIZE)
		#self.verbose =  1

		self.Trainer = self.training()    
		
	

	def training(self):
		dataLoader = DataLoader(self.filepath)		
		images, labels = dataLoader.create_dataset()
		#images_validation, labels_validation = self.loadData(self.validation_dataset_filepath)
		# Print the batch number at the beginning of every batch.
		
		callbacks = Cback()
		callbacks = callbacks.makeCb()
		
		self.model.fit(x=images, y=labels, steps_per_epoch=1, epochs=1, callbacks= callbacks)
		return self.model
		
	
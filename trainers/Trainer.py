import sys, os
parent_dir = os.getcwd()
sys.path.append("/home/ip/IPNeuronaleNetze")
import tensorflow as tf

from models.optimizer.cb import Cback
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.applications.vgg16 import preprocess_input
import numpy as np
import pandas as pd



class Trainer:
	
	def __init__(self, model, filepath, validation_filepath, test_filepath, identifier, epochs = 312):		
		self.filepath = filepath
		self.model = model		
		self.saved_model_path = ""
		self.identifier = identifier
		self.validation_filepath = validation_filepath
		self.test_filepath = test_filepath		
		self.epochs = epochs
		self.Trainer = self.train()

	def train(self):
		
		# Set batch_size for training
		batch_size = 12

		# Set settings for saving the model weights after training 
		# and the class_mode depending on gender or age estimation
		if self.identifier == 1:
			self.saved_model_path = os.getcwd()+ "/models/GenderWeights"
			weights_name = "model_gender.h5"			
			class_mode ='binary'
		else:
			self.saved_model_path = os.getcwd() + "/models/AgeWeights"
			weights_name = "model_age.h5"			
			class_mode = 'categorical'			
	
		# Setting up the the generators for training, using the keras preprocessing_function
		datagen = ImageDataGenerator(preprocessing_function = preprocess_input)			
			
		train_generator = datagen.flow_from_directory(
				self.filepath, target_size = (224,224), 
				batch_size=batch_size, class_mode = class_mode, 
				shuffle = True)
                
		validation_generator = datagen.flow_from_directory(
				self.validation_filepath, target_size =(224,224),
				batch_size = batch_size, class_mode = class_mode, 
				shuffle = True)
		
		test_generator = datagen.flow_from_directory(
				self.test_filepath, target_size =(224,224), 
				batch_size = 1,	class_mode = class_mode, 
				shuffle = False)			

		# Loading the callbacks defined in the cb.py	
		callbacks = Cback()
		callbacks = callbacks.makeCb()
		
		# Start the training
		history = self.model.fit_generator(
				train_generator, steps_per_epoch = train_generator.samples/train_generator.batch_size, 
				validation_data=validation_generator, validation_steps = validation_generator.samples/validation_generator.batch_size, 
				epochs = self.epochs , callbacks = callbacks)

		# Save weights in h5 format				
		self.model.save_weights(weights_name)

		# Save weights in saved_model format
		self.saved_model_path = tf.contrib.saved_model.save_keras_model(
            	self.model, self.saved_model_path, 
				custom_objects=None, as_text=None)
		
		# Evaluate the trained model and print results
		scores = self.model.evaluate_generator(
				test_generator,test_generator.samples)

		print(
				"Evaluation Loss: ", scores[0], 
				"Evaluation MAE: ", scores[1])

		# Reset the test_generator for predictions
		test_generator.reset()
		# Make predictions on test_generator
		pred = self.model.predict_generator(
				test_generator, verbose=1)	

		# Write prediction results in separate .csv file
		self.write_results(
				pred, test_generator)
		
		# Return the trained model
		return self.model		

	def write_results(self, predictions, test_generator):
		# Set settings for either age or gender estimation
		# Gender: get rounded value: 0 or 1
		if self.identifier == 1:
			predicted_class_indices = np.around(predictions)
			results_name = "results_imdb_gender.csv"
		# Age: get index with max value
		else:
			predicted_class_indices = np.argmax(
				predictions, axis=1)
			results_name = "results_imdb_age.csv"

		# Get the matching labels from test_generator to write them into .csv file
		labels = (test_generator.class_indices)
		labels = dict((v,k) for k,v in labels.items())
		predictions = [labels[k] for k in predicted_class_indices[:,0]]
		filenames = test_generator.filenames
		results = pd.DataFrame ({
			"Filename":filenames,
			"Predictions":predictions})
		results.to_csv(results_name, index = False)	


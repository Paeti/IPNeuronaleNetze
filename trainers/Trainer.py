import sys
sys.path.append("/IPNeuronaleNetze")
import tensorflow as tf
from tensorflow.python.keras import callbacks
from models.optimizer.cb import Cback
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.applications.vgg16 import preprocess_input
import numpy as np
import pandas as pd

class Trainer:
        def __init__(self, model, filepath, validation_filepath, test_filepath, identifier, epochs = 312, save_model = True):
            self.filepath = filepath
            self.model = model
            self.saved_model_path = ""
            self.identifier = identifier
            self.validation_filepath = validation_filepath
            self.test_filepath = test_filepath
            self.epochs = epochs
            self.save_model = save_model
	    

        def train(self):
		# Set batchsize, the higher, the faster the training
		# Steps per epoch will later then be calculated automatically by diving samples/batch_size
		batch_size = 42

		# Set filepath where our model will be saved.
		# Please add directory to this project in the beginning of the string
		if self.identifier == 1:
				self.saved_model_path = "/IPNeuronaleNetze/models/GenderWeights"
				class_mode ='binary'
		else:
				self.saved_model_path = "/IPNeuronaleNetze/models/AgeWeights"                        
				class_mode = 'categorical'

		# Load the datasets via ImageDataGenerator
		datagen = ImageDataGenerator(preprocessing_function = preprocess_input)

		train_generator = datagen.flow_from_directory(
				self.filepath, target_size = (224,224),
				batch_size=batch_size, class_mode = class_mode, 
				shuffle = True)

		validation_generator = datagen.flow_from_directory(
				self.validation_filepath, target_size =(224,224),
				batch_size = batch_size, class_mode = class_mode, 
				shuffle = True)

		# Load the callbacks, which are separated in a different class
		callbacks = Cback()
		callbacks = callbacks.makeCb() 

		# Start the training wit the fit_generator
		history = self.model.fit_generator(
				train_generator, steps_per_epoch = train_generator.samples/train_generator.batch_size,
				validation_data=validation_generator, validation_steps = validation_generator.samples/validation_generator.batch_size, 
				epochs = self.epochs ,callbacks = callbacks)                                                                             

		# Save the model if demanded																												
		if self.save_model == True:
				self.saved_model_path = tf.contrib.saved_model.save_keras_model(                                      
						self.model, self.saved_model_path, 
						custom_objects=None, as_text=None)  

		# Evaluate the trained model
		self.evaluate(self.test_filepath)   

		return self.model
                                                                                                                              
                                                                                                                              
        def evaluate(self, test_filepath, name = "evaluation"):
				# Set the right parameters for evaluation
                if self.identifier == 1:
                        class_mode ='binary'
                        results_name = "_gender.csv"
                else:
                        class_mode = 'categorical'
                        results_name = "_age.csv"
                # Load the test dataset via ImageDataGenerator                                                                                                        
                datagen = ImageDataGenerator(preprocessing_function = preprocess_input)                                                                                                                                                                    
                test_generator = datagen.flow_from_directory(
						test_filepath, target_size =(224,224), 
						batch_size = 1, class_mode = class_mode, 
						shuffle = False)

                # Start the prediction walk through                                                                                                           
                pred = self.model.predict_generator(test_generator, verbose=1)

                # Depending on gender or age estimation we need our class_indices in a different structure  
				# for the predictions                                                                                                            
                if self.identifier == 1:
                        predicted_class_indices = np.around(pred)                                                             
                else:
                        predicted_class_indices = np.argmax(pred, axis=1)
                                                                                                                              
                labels = (test_generator.class_indices)                                                                       
                labels = dict((v,k) for k,v in labels.items())
                                                                                                                              
                if self.identifier == 1:
                        predictions = [labels[k] for k in predicted_class_indices[:,0]]
                else:
                        predictions = [labels[k] for k in predicted_class_indices]
                                                                                                                              
                filenames = test_generator.filenames 
				# Create the .csv file for the evaluation                                                                         
                results = pd.DataFrame ({"Filename":filenames,
										"Predictions":predictions})
                results.to_csv(name+results_name, index = False)
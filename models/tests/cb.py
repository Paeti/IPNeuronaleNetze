import tensorflow as tf
from tensorflow import keras 
from DataLoader import DataLoader
from keras import callbacks
import json


class Cback:
	
	def __init__(self):
		self.batchsize =  []    
		
	

	def makeCb(self):
		
		#images_validation, labels_validation = self.loadData(self.validation_dataset_filepath)
		# Print the batch number at the beginning of every batch.
		batch_print_callback = keras.callbacks.LambdaCallback(
			on_batch_begin=lambda batch,logs: print(batch))

		# Stream the epoch loss to a file in JSON format. The file content
		# is not well-formed JSON but rather has a JSON object per line.
		
		json_log = open('loss_log.json', mode='wt', buffering=1)
		json_logging_callback = keras.callbacks.LambdaCallback(
			on_epoch_end=lambda epoch, logs: json_log.write(
				json.dumps({'epoch': epoch, 'loss': logs['loss']}) + '\n'),
			on_train_end=lambda logs: json_log.close()
		)

		# Terminate some processes after having finished model training.
		processes = ...
		cleanup_callback = keras.callbacks.LambdaCallback(
			on_train_end=lambda logs: [
				p.terminate() for p in processes if p.is_alive()])
		
		callback =[keras.callbacks.EarlyStopping(monitor='loss', min_delta=0, patience=2, verbose=0, mode='auto', baseline=None), batch_print_callback,json_logging_callback]
		
		return callback
		
		
	
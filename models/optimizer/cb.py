import sys
sys.path.append("/home/ip/IPNeuronaleNetze")
from time import time
import tensorflow as tf
import tensorflow.python.keras
from tensorflow.python.keras import callbacks
import json


class Cback:

        def __init__(self):                
			self.batchsize = []

        def makeCb(self):
                #images_validation, labels_validation = self.loadData(self.validation_dataset_filepath)
                # Print the batch number at the beginning of every batch.
                trainedbatch = []
                batch_print_callback = callbacks.LambdaCallback(
                on_batch_begin=lambda batch,logs: trainedbatch.append(batch))

                # Stream the epoch loss to a file in JSON format. The file content
                # is not well-formed JSON but rather has a JSON object per line.

                json_log = open('loss_log.json', mode='wt', buffering=1)
                json_logging_callback = callbacks.LambdaCallback(
                        on_epoch_end=lambda epoch, logs: json_log.write(
                                json.dumps({'epoch': epoch, 'loss': logs['loss'], 
											'mae': logs['mean_absolute_error'], 'val_loss':logs['val_loss'],
											'val_mae':logs['val_mean_absolute_error']}) + '\n'),
                        on_train_end=lambda logs: json_log.close()
                )

                # Terminate some processes after having finished model training.
                processes = []
                cleanup_callback = callbacks.LambdaCallback(
                        on_train_end=lambda logs: [
                                p.terminate() for p in processes if p.is_alive()])

                tensorboard = callbacks.TensorBoard(log_dir="logs/{}".format(time()))

                callback =[callbacks.EarlyStopping(monitor='val_loss', patience=4, mode = 'auto'), batch_print_callback,
							json_logging_callback, cleanup_callback]
			
		return callback
		
		
	

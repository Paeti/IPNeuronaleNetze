import tensorflow as tf

from tensorflow import keras 
#from keras import callbacks
import json


class Trainer:
    def __init__(self, model, training_dataset_filepath, validation_dataset_filepath, saved_model_path):
      self.model = model
      self.training_dataset_filepath = training_dataset_filepath
      self.saved_model_path = saved_model_path
      self.batchsize =  100
      self.epochs =  50
      self.validation_dataset_filepath =  validation_dataset_filepath
      #self.verbose =  1

    def loadData(self, filepath):
        feature = {'train/image': tf.FixedLenFeature([], tf.string),
               'train/label': tf.FixedLenFeature([], tf.int64)}
        # Create a list of filenames and pass it to a queue
        filename_queue = tf.train.string_input_producer(
            [filepath], num_epochs=1)
        # Define a reader and read the next record
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        # Decode the record read by the reader
        features = tf.parse_single_example(
            serialized_example, features=feature)
        # Convert the image data from string back to the numbers
        images = tf.decode_raw(features['train/image'], tf.float32)

        # Cast label data into int32
        labels = tf.cast(features['train/label'], tf.int32)
        # Reshape image data into the original shape
        images = tf.reshape(images, [224, 224, 3])
        return images, labels

    def training(self):
        images_training, labels_training = self.loadData(self.training_dataset_filepath)
        images_validation, labels_validation = self.loadData(self.validation_dataset_filepath)
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




        earlyStopping_callback = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto', baseline=None)

        self.model.fit(images_training, labels_training, batchsize=self.batchsize, epochs=self.epochs, validation_data=(images_validation, labels_validation), callbacks=[earlyStopping_callback, batch_print_callback, json_logging_callback, cleanup_callback])
        # validation_data=testing_set.make_one_shot_iterator(),validation_steps=len(x_test) // _BATCH_SIZE,verbose = 1)

        tf.contrib.saved_model.save_keras_model(
            self.model, self.saved_model_path, custom_objects=None, as_text=None)

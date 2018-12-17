import tensorflow as tf


class Trainer:
    def __init__(self, model, filepath, saved_model_path):
      self.model = model
      self.filepath = filepath
      self.saved_model_path = saved_model_path
      self.batchsize =  # TODO
      self.epochs =  # TODO
      self.validation_data =  # TODO
      self.verbose =  # TODO

    def loadData(filepath):
        feature = {'train/image': tf.FixedLenFeature([], tf.string),
               'train/label': tf.FixedLenFeature([], tf.int64)}
        # Create a list of filenames and pass it to a queue
        filename_queue = tf.train.string_input_producer(
            [data_path], num_epochs=1)
        # Define a reader and read the next record
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        # Decode the record read by the reader
        features = tf.parse_single_example(
            serialized_example, features=feature)
        # Convert the image data from string back to the numbers
        image = tf.decode_raw(features['train/image'], tf.float32)

        # Cast label data into int32
        label = tf.cast(features['train/label'], tf.int32)
        # Reshape image data into the original shape
        image = tf.reshape(image, [224, 224, 3])
        return images, labels

    def training():
        images, labels = loadData(self.filepath)
        self.model.fit(images, labels, batchsize=self.batchsize, epochs=self.epochs,  # TODO)
        # validation_data=testing_set.make_one_shot_iterator(),validation_steps=len(x_test) // _BATCH_SIZE,verbose = 1)

        tf.contrib.saved_model.save_keras_model(
            model, saved_model_path, custom_objects=None, as_text=None)

import tensorflow as tf

from tensorflow import keras 
#from keras import callbacks
import json

class DataLoader:
    def __init__(self, filepath):  
        self.filepath  = filepath              
        self.batchsize =  100
        self.epochs =  50
        #self.validation_dataset_filepath =  validation_dataset_filepath
        self.steps_per_epoch = 10000/self.batchsize #Noch anpassen (SUM_OF_ALL_DATASAMPLES / BATCHSIZE)
        #self.verbose =  1       

    

    def _parse_function(self,proto):
        # define your tfrecord again. Remember that you saved your image as a string.
        keys_to_features = {'train/image': tf.FixedLenFeature([], tf.string),
                            "train/label": tf.FixedLenFeature([], tf.int64)}
        
        # Load one example
        parsed_features = tf.parse_single_example(proto, keys_to_features)
        
        # Turn your saved image string into an array
        parsed_features['train/image'] = tf.decode_raw(
            parsed_features['train/image'], tf.float32)
        
        return parsed_features['train/image'], parsed_features["train/label"]

    
    def create_dataset(self):
        
        # This works with arrays as well
        dataset = tf.data.TFRecordDataset(self.filepath)
        
        # Maps the parser on every filepath in the array. You can set the number of parallel loaders here
        dataset = dataset.map(self._parse_function, num_parallel_calls=1)
        
        # This dataset will go on forever
        dataset = dataset.repeat()
        
        # Set the number of datapoints you want to load and shuffle 
        dataset = dataset.shuffle(1)
        
        # Set the batchsize
        dataset = dataset.batch(1)
        
        # Create an iterator
        iterator = dataset.make_one_shot_iterator()
        
        # Create your tf representation of the iterator
        image, label = iterator.get_next()

        # Bring your picture back in shape
        image = tf.reshape(image, [-1, 224, 224, 3])
        
        # Create a one hot array for your labels
        label = tf.one_hot(label, 1)
        
        return image, label


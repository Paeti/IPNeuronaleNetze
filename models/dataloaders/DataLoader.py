import tensorflow as tf

import tensorflow.python.keras
from tensorflow.python.keras import callbacks
import json
import numpy as np

class DataLoader:

    def __init__(self, filepath, identifier):  
        self.filepath = filepath  
        self.identifier = identifier	
		

    def _parse_function(self,proto):
        # define your tfrecord again. Remember that you saved your image as a string.
        keys_to_features = {'train/image': tf.FixedLenFeature([], tf.string),
                            'train/label': tf.FixedLenFeature([], tf.int64)}
        
        # Load one example
        parsed_features = tf.parse_single_example(proto, keys_to_features)        

        # Turn your saved image string into an array
        parsed_features['train/image'] = tf.decode_raw(
            parsed_features['train/image'], tf.float32)
        
        return parsed_features['train/image'], parsed_features["train/label"]

    
    def create_dataset(self,buffer_size = 2048, train = True, batch_size = 1 ):
        
        # This works with arrays as well
        dataset = tf.data.TFRecordDataset(self.filepath)
        
        # Maps the parser on every filepath in the array. You can set the number of parallel loaders here
        dataset = dataset.map(self._parse_function)
        
        if train:
        # If training then read a buffer of the given size and                    

        # Allow infinite reading of the data.
            num_repeat = None
        else:
        # If testing then don't shuffle the data.        
        # Only go through the data once.
            num_repeat = 1

        # Repeat the dataset the given number of times.
        dataset = dataset.repeat()       
        
        # Set the number of datapoints you want to load and shuffle        
        dataset = dataset.shuffle(2048)

        # Set the batchsize
        dataset = dataset.batch(batch_size)       
        
        # Create an iterator
        iterator = dataset.make_one_shot_iterator()
        
        # Create your tf representation of the iterator
        #image, label = iterator.get_next()
         
        next_element = iterator.get_next()      


        with tf.Session() as sess:
            try:
                i=0
                while i<10:
                    data_record = sess.run(next_element)
                    print(data_record[0].shape)
                    print(type(data_record))
                    print(type(data_record[0]))
                    #data_record[0] = np.reshape(data_record[0],(6,))
                    print("----------------------------------------------------------")
                    image = data_record[0]
                    print(image)
                    print(image.dtype)
                    img_mat = image.reshape(-224,224,3)
                    print(img_mat.shape)

                    if i == 5:
                        np.save('test2.out',image)

                    i=i+1
            except:
                pass 

        #Bring your picture back in shape
        #image = tf.reshape(image, [-1, 224, 224, 3])
	#image = tf.subtract(image, 116.779)
	#image = tf.reverse(image, axis= [2])
        
        if self.identifier == 1 :
             #Create a one hot array for your labels
             label = tf.one_hot(label, 1)
        else:
             #Create a one hot array for your labels
             label = tf.one_hot(label, 101)
     
        #print("__________________________________sdfjksdfjkdshfsdhfsdfhdsfj__________________________________________")
        #print(image)   
        #print("------------------------------------------------------------dfuhdsfhshf--------------------------")
        #print(label)
        #print("----------------------------------------------------------------------------")
        #print(dataset.output_shapes)   
        return image, label


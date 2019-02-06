The OurModel class builds and compiles the convolutional networks for gender or age estimation.
Depending on the identifier in the construcor either a gender or a age estimation network will be build.

The model from the OurModel class consists of a vgg16 base model with the pre-trained weights from 'imagenet' and 5 additional layers.

So the structure is as follows:
    
    16 Layer from the vgg16 base model    
    1 GlobalAveragePooling2D layer
    2 Dense layer

    1 Dense layer with sigmoid for age estimation
        or
    1 Dense layer with softmax for gender estimation

The structure of this model is inspired by the following paper.
    https://www.vision.ee.ethz.ch/en/publications/papers/proceedings/eth_biwi_01229.pdf

As optimizer the GradientDescentOptimizer from tf.train is use.
This optimizer makes it possible to load and save the trained model as saved_model file effortlessly. 
    https://www.tensorflow.org/api_docs/python/tf/train/GradientDescentOptimizer


The load_model method offers the possibility to load a saved_model file and compile it allready. 
So after the model is loaded with this method, it's ready for training.



    






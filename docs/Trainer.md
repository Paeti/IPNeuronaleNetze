The Trainer class does the training for gender or age estimation depending on the identifier given in the constructor of this class.

Loading and preprocessing of the dataset will be done by the ImageDataGenerator from keras.
All the preprocessing parameters are adjusted by the preprocess_input from keras. These are the parameters which were used for the imagent weights.

The Trainer class takes use of the Cback class, which consits of the callbacks for the training. 
These callbacks are in use: 
    EarlyStopping
    batch_print_callback
    json_logging_callback
    cleanup_callback
    
For further reading:
https://keras.io/callbacks/

The train method returns the trained model and saves the model in the Trainer instance.
The trained model will be saved per default. This will be default directory for the saved model:
    
    /IPNeuronaleNetze/models/GenderWeights for gender
        or
    /IPNeuronaleNetze/models/AgeWeights" for age

A .csv file will be created by the evaluate method. Use this .csv file in combination with the parser scripts to interpret the results of the training.





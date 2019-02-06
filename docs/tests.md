The modelTest class offers a few methods to confirm the proper functionality of the convolutional network you want to train.
It is advisable to take a small dataset with just a couple of hundreds of pictures. Due to the test environment each single step
of training will take more ressources of your computer as well as time than it would normally do.

This testclass tests the correct structure of the OurModel class, which will be used for training.
Also it tests the training itself, to confirm the right functionality within the training. 

Test cases:
    1. Test whether saved model file will be created after training
        test_model_get_saved
    
    2. Test whether model will be saved and loaded properly
        test_model_get_saved_and_loaded_correctly

    3. Test whether the weigts are changed after one step of training
        test_one_training_step

    4. Test whether all of the layers of a model are trainable, except the input layers
        test_model_layers_allTrainable

    5. Test whether Gendermodel is in expected shape
        test_Gendermodel_layerLength 

    6. Test whether Gendermodel is in expected shape
        test_Agemodel_layerLength

Recommendation:
    Use the test.py before the actual training.

Note: 
    Test case 1 and 2 will create a folder with a saved_model file.
    You might want to delete these folders before the actual training. 

Inspiration for the test cases:
    https://medium.com/@keeper6928/how-to-unit-test-machine-learning-code-57cf6fd81765

    
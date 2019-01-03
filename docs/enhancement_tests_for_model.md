Testing whether the model is working correctly
within modelTest class

Research: https://medium.com/@keeper6928/mltest-automatically-test-neural-network-models-in-one-function-call-eb6f1fa5019d


Tests:

- test_weights_get_saved: Test whether all the weights of our model get saved           correctly after training, by checking the path of the new build weight file.

- test_one_training_step: Compare the weights of the model before and after one         step of training. If there is a difference this test is passed.

- test_Model_layers_allTrainable: Test whether all of the layers of a model are         trainable, except the input layers

- test_GenderModel_layerLength/ test_AgeModel_layerLength: Test wether the models       layer length are in expected manner

- test_loss_AgeModel/test_loss_GenderModel: Test wether the loss of the models is not   equal 0





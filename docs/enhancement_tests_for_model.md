Testing whether the model is working correctly
Using Unittest and make use of "mltest"

mltest: Designed for testing neural network models 
Research: https://medium.com/@keeper6928/mltest-automatically-test-neural-network-models-in-one-function-call-eb6f1fa5019d
Source: https://github.com/Thenerdstation/mltest

Tests:

- test_convent: Test whether all variables created get trained
- test_loss: Test whether our loss is 0 
- test_gen_training: Test that only the variables we want to train actually get trained
- drop_out_test: Test the dropout probability to a given value
- accuracy_test: Test correctness of accuracy function
- shape_test: Test that tensor is in expected shape

- mltest: 



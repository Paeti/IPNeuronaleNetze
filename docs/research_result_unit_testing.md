Needed librarys:
    Unittest : Make use of "unittest.TestCase"Class

    Consider to use mltest library:
    Benefits:
    Test suite for machine learning in tensorflow with
    Allready implemented test functions for tensors
    Easy and quick to use

Recommendations for test patterns:

    Keep them deterministic
    Keep the tests short

Example structure for a unit test:
    class Test(unittest.TestCase):
        def test_1(self):
        self.assertEqual(True, True)

Example unit test for a given accuracy function:
    class accuracy_test(tf.test.TestCase):
        def accuracy_exact_test(self):
            with self.test_session():
         	    test_preds 0 [[0.9, 0.1], [0.01 , 0.99]] 
         	    test_targets = [0, 1]
         	    test_acc = get_accuracy(test_preds, test_targets)
         	    self.assertEqual(test_acc.eval(), 100)

Sources:
    unit testing
    https://guillaumegenthial.github.io/testing.html

    mltest
    https://github.com/Thenerdstation/mltest/blob/master/README.md
    https://medium.com/@keeper6928/mltest-automatically-test-neural-network-models-in-one-function-call-eb6f1fa5019d

Note:
    Test nn not yet implemented at this point.
    Example test may be added afterwards.
import tensorflow as tf
import unittest
import numpy as np
import mltest
#import your_model_file

#Test whether all variables created get trained
def test_convnet():
    image = tf.placeholder(tf.float32, (None, 100, 100, 3)
    model = Model(image)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    before = sess.run(tf.trainable_variables())
    _ = sess.run(model.train, feed_dict={
            image: np.ones((1, 100, 100, 3)),
            })
    after = sess.run(tf.trainable_variables())
    for b, a, n in zip(before, after):
        # Make sure something changed.
        assert (b != a).any()

#Test whether our loss is 0
def test_loss():
    in_tensor = tf.placeholder(tf.float32, (None, 3))
    labels = tf.placeholder(tf.int32, None, 1))
    model = Model(in_tensor, labels)
    sess = tf.Session()
    loss = sess.run(model.loss, feed_dict={
    in_tensor:np.ones(1, 3),
    labels:[[1]]
    })
    assert loss != 0

#Test that only the variables we want to train actually get trained
def test_gen_training():
    model = Model
    sess = tf.Session()
    gen_vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope='gen')
    des_vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope='des')
    before_gen = sess.run(gen_vars)
    before_des = sess.run(des_vars)
    # Train the generator.
    sess.run(model.train_gen)
    after_gen = sess.run(gen_vars)
    after_des = sess.run(des_vars)
    # Make sure the generator variables changed.
    for b,a in zip(before_gen, after_gen):
      assert (a != b).any()
    # Make sure descriminator did NOT change.
    for b,a in zip(before_des, after_des):
      assert (a == b).all()

#Test that dropout probability is grater than !X so that the model is not changed to attempt to train are more then !X dropout
class drop_out_test(tf.test.TestCase):
    def dropout_greaterthan(self):
        with self.test_session():
            self.assertGreater(dropout.eval(), X)

#Test wheter accuracy function behaves properly
class accuracy_test(tf.test.TestCase):
    def accuracy_exact_test(self):
        with self.test_session():
            test_preds = [[0.9, 0.1], [0.01, 0.99] 
            test_targets = [0, 1]
            test_acc = get_accuracy(test_preds, test_targets) 
            self.assertEqual(test_acc.eval(), 100)

#Test whether tensor is in expected shape
class shape_test(tf.test.TestCase):
    def output_shape_test(self):
        with self.test_session():
            numpy_array = np.ones([batch_size, target_size])
            self.assertShape(numpy_array, model_output)


#____main_____



#Use mltest
def test_my_model():  
    # Make placeholders for input into the model
    input_tensor = tf.placeholder(tf.float32, (None, 100))
    label_tensor = tf.placeholder(tf.int32, (None))
    # Build your model.
    model = your_model_file.build_model(input_tensor, label_tensor)
    # Give it some random input (Be sure to seed it!!).
    feed_dict = {
        input_tensor: np.random.normal(size=(10, 100)),
        label_tensor: np.random.randint((100))
    }
# Run the test suite!
mltest.test_suite(
    model.prediction,
    model.train_op,
    feed_dict=feed_dict)  
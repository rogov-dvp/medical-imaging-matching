import unittest
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from CNN import train_cnn
 
class Test_CNN(unittest.TestCase):
    def test_input_shape(self): 
        """
        Test that the input shape is as expected
        """
        shape = ()
        image = tf.ones(shape)
        self.unet.build()
        self.assertEqual(self.unet.model.predict(image).shape, shape)

    def test_output_shape(self):
        """
        Test that the output shape of CNN is as expected
        """
        shape = (1, self.unet.image_size, self.unet.image_size, 3)
        image = tf.ones(shape)
        self.unet.build()
        self.assertEqual(self.unet.model.predict(image).shape, shape)

    def test_weights_are_changing(self):
        """
        Tests that the CNN actually learns
        """
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

    def test_loss_decreases(self):
        """
        Tests that the loss generally decreases
        """
        in_tensor = tf.placeholder(tf.float32, (None, 3))
        labels = tf.placeholder(tf.int32, None, 1)
        model = Model(in_tensor, labels)
        sess = tf.Session()
        loss = sess.run(model.loss, feed_dict={
            in_tensor:np.ones(1, 3),
            labels:[[1]]
        })
  

        self.assertTrue(value != 0)

    def test_loss_is_never_zero(self):
        """
        Tests that the loss does no get zero
        """
        in_tensor = tf.placeholder(tf.float32, (None, 3))
        labels = tf.placeholder(tf.int32, None, 1)
        model = Model(in_tensor, labels)
        sess = tf.Session()
        loss = sess.run(model.loss, feed_dict={
            in_tensor:np.ones(1, 3),
            labels:[[1]]
        })
        assert loss != 0
    
    def test_trained(self):
        """
        Tests that the model can be trained
        """
    self.assertEqual()

if __name__ == "__main__":
    unittest.main()
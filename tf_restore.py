'''
Created on Aug 18, 2017

@author: kashefy
'''
import os
import shutil
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

if __name__ == '__main__':
    validate_only = True # Switch to True after first trainign run, write down the final values of the weights for comparison
    if not validate_only:
        if os.path.isdir('./a'):
            shutil.rmtree('a')
        os.makedirs('./a')
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    name_w = 'W'
    
    x = tf.placeholder(tf.float32, [None, 784])
    with tf.variable_scope("var_scope", reuse=None):
        W = tf.get_variable(name_w, shape=[784, 10],
                            initializer=tf.random_normal_initializer(stddev=0.1))
        b = tf.get_variable('b', shape=[10],
                            initializer=tf.constant_initializer(0.1))
    logits = tf.matmul(x, W) + b
    y = tf.nn.softmax(logits)
    y_ = tf.placeholder(tf.float32, [None, 10])
    cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(\
                        labels=y_, logits=logits))
    train_step = tf.train.GradientDescentOptimizer(0.9).minimize(cross_entropy)
    print [op.name for op in tf.get_default_graph().get_operations() if op.op_def and 'Variable' in op.op_def.name]

    init_op = tf.global_variables_initializer()
        
    with tf.Session() as sess:
        sess.run(init_op)
        if not validate_only:
            saver = tf.train.Saver(max_to_keep=5)
        else:
            saver = tf.train.import_meta_graph('./a/x-999.meta')
            saver.restore(sess, tf.train.latest_checkpoint('./a/'))
        print [op.name for op in tf.get_default_graph().get_operations() if op.op_def and 'Variable' in op.op_def.name]

        w0 = np.copy(sess.run(W))
        print(sess.run(W).flatten()[406:412])
        for itr in range(1000):
            batch_xs, batch_ys = mnist.train.next_batch(128)
            _, c = sess.run([train_step, cross_entropy],
                     feed_dict={x: batch_xs, y_: batch_ys})
            #print(itr, c)
        print(sess.run(W).flatten()[406:412])
        print np.array_equal(w0, sess.run(W))
        
        if not validate_only:
            saver.save(sess, './a/x', global_step=itr)
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print(sess.run(accuracy,
                       feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
        print(sess.run(accuracy,
                       feed_dict={x: mnist.validation.images, y_: mnist.validation.labels}))
    pass

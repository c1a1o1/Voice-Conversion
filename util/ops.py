import tensorflow as tf
import numpy as np
def _batch_norm(input, is_training):
    """ Batch Normalization
    """
    with tf.variable_scope("batch_norm"):
        return tf.contrib.layers.batch_norm(input,
                                        decay=0.9,
                                        scale=True,
                                        updates_collections=None,
                                        is_training=is_training)

def nchw_to_nhwc(x):
    return tf.transpose(x, [0, 2, 3, 1])

def nhwc_to_nchw(x):
    return tf.transpose(x, [0, 3, 1, 2])

def _leaky_relu(input, slope=0.3):
    return tf.maximum(slope*input, input)

def _instance_norm(input):
    """ Instance Normalization
    """
    with tf.variable_scope("instance_norm"):
        depth = input.get_shape()[3]
        scale = _weights("scale", [depth], mean=1.0)
        offset = _biases("offset", [depth])
        mean, variance = tf.nn.moments(input, axes=[1,2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (input-mean)*inv
        return scale*normalized + offset

def _biases(name, shape, constant=0.0):
    """ Helper to create an initialized Bias with constant
    """
    return tf.get_variable(name, shape,
            initializer=tf.constant_initializer(constant))

def conv2d(input_, output_dim, kernel, stride, data_format='NCHW', stddev=0.02, norm=None, reuse=False,name="conv2d"):
    with tf.variable_scope(name, reuse=reuse):
        input_ = nchw_to_nhwc(input_)
        w = tf.get_variable('w', [kernel[0], kernel[1], input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, stride[0], stride[1], 1], padding='SAME')#, data_format=data_format)
        print "conv shape:", conv.get_shape().as_list()
        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        shape = [int(input_.get_shape()[0], tf.int32),
        conv.get_shape().as_list()[1], conv.get_shape().as_list()[2], conv.get_shape().as_list()[3]]
        print "shape:", shape
        conv = tf.reshape(tf.nn.bias_add(conv, biases), tf.cast(shape, tf.int32))
        if norm == "batch_norm":
            conv = _batch_norm(conv, True)
        elif norm == 'instance_norm':
            conv = _instance_norm(conv)
        conv = nhwc_to_nchw(conv)
        print "output conv shape:", conv.get_shape().as_list()
        return conv

def deconv2d(input_, output_shape, kernel, stride, data_format="NHWC", stddev=0.02, norm=False, reuse=None,
             name="deconv2d", with_w=False):
    with tf.variable_scope(name, reuse=reuse):
        # filter : [height, width, output_channels, in_channels]
        input_ = nchw_to_nhwc(input_)
        print "input_ shape:", input_.get_shape().as_list()
        w = tf.get_variable('w', [kernel[0], kernel[1], output_shape, input_.get_shape().as_list()[-1] ],
                            initializer=tf.random_normal_initializer(stddev=stddev))

        shape = [int(input_.get_shape().as_list()[0]),
        input_.get_shape().as_list()[1]*stride[0], input_.get_shape().as_list()[2]*stride[1], output_shape]
        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=tf.cast(shape, tf.int32),
                                strides=[1, stride[0], stride[1], 1])
            print "deconv shape:", deconv.get_shape().as_list()
        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,data_format=data_format,
                                strides=[1, stride[0], stride[1], 1])

        biases = tf.get_variable('biases', [output_shape], initializer=tf.constant_initializer(0.0))
        shape = [int(input_.get_shape().as_list()[0]),
        deconv.get_shape().as_list()[1], deconv.get_shape().as_list()[2], deconv.get_shape().as_list()[3]]
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), tf.cast(shape, tf.int32))
        if norm == "batch_norm":
            deconv = _batch_norm(deconv, True)
            deconv = nhwc_to_nchw(deconv)
        elif norm == 'instance_norm':
            deconv = _instance_norm(deconv)
        if with_w:
            return deconv, w, biases
        else:
            return deconv

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False,
    name="linear", reuse=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(name, reuse=reuse) as scope:
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias

def _weights(name, shape, mean=0.0, stddev=0.02):
    """ Helper to create an initialized Variable
    Args:
    name: name of the variable
    shape: list of ints
    mean: mean of a Gaussian
    stddev: standard deviation of a Gaussian
    Returns:
    A trainable variable
    """
    var = tf.get_variable(
    name, shape,
    initializer=tf.random_normal_initializer(
      mean=mean, stddev=stddev, dtype=tf.float32))
    return var

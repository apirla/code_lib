# there are some models of network
import tensorflow as tf
import math


def weight_variable(shape,trainable = True,f_in = None,f_out = None):
    """weight_variable generates a weight variable of a given shape."""
    if f_in and f_out:#若有f_in、f_out 输入,Xavier权重初始化
        initial = tf.random_uniform(shape,minval=-math.sqrt(6.0/(f_in+f_out)),maxval=math.sqrt(6.0/(f_in+f_out)),dtype=tf.float32)
    else:
        initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial,trainable=trainable)
def bias_variable(shape,trainable = True):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial,trainable=trainable)
def fc( name, input_data, out_channel, trainable=True):
    """
    全相连层
    :param name: 
    :param input_data: a tensor
    :param out_channel:  下一层的输入数
    :param trainable:  
    :return: a tensor
    """
    shape = input_data.get_shape().as_list()
    if len(shape) == 4:
        size = shape[-1] * shape[-2] * shape[-3]
    else:
        size = shape[1]
    input_data_flat = tf.reshape(input_data, [-1, size])
    with tf.variable_scope(name):
        weights = weight_variable(shape=[size, out_channel],trainable=trainable)
        biases = bias_variable(shape = [out_channel],trainable=trainable)
        res = tf.matmul(input_data_flat, weights)
        out = tf.nn.relu(tf.nn.bias_add(res, biases))
    return out
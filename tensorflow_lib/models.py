# there are some models of network
import tensorflow as tf
import math


def weight_variable(shape,trainable = True,f_in = None,f_out = None,name = None):
    '''
    
    :param shape:  a list that describe the shape of tensor
    :param trainable: 
    :param f_in: 
    :param f_out: 
    :return: a variable
    '''
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
def maxpool(name, input_data ,height = 2, width = 2 ,strides = 2):
    """
    
    :param name: 
    :param input_data: a tensor
    :param height: pool 's height
    :param width:  pool 's width
    :param strides: step len
    :return: a tensor 
    """
    out = tf.nn.max_pool(input_data, [1,height, width, 1], [1, strides, strides, 1], padding="SAME", name=name)
    return out
def conv_layer(name, input_data, out_channel, trainable = True,strides = 1):
    '''
    卷积层
    :param name: 卷积层名
    :param input_data: 输入的张量
    :param out_channel: 输出的轨道数
    :param trainable: 是否训练？
    :param strides: 步长
    :return: 一个张量
    '''
    in_channel = input_data.get_shape()[-1]  # 从input张量中推测输入道数
    with tf.variable_scope(name):
        kernel = weight_variable([3, 3, in_channel, out_channel],trainable=trainable)
            # tf.get_variable("weights", [3, 3, in_channel, out_channel], dtype=tf.float32,
            #                      trainable=trainable)  # 权重
        biases = bias_variable([out_channel],trainable=trainable)
            # tf.get_variable("biases", [out_channel], dtype=tf.float32, trainable=trainable)
        conv_res = tf.nn.conv2d(input_data, kernel, [1, strides, strides, 1], padding="SAME")
        res = tf.nn.bias_add(conv_res, biases)
        out = tf.nn.relu(res, name=name)
    return out
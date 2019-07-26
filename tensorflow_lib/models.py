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
def identity_block( X_input, kernel_size, in_filter, out_filters, stage, block, training,trainable = True):
    """
    三层卷积残差块，无升降维
    !!! in_filter must the same as out_filters
    Implementation of the identity block as defined in Figure 3
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    kernel_size -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    training -- train or test for bn
    trainable -- train or test for filters
    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    block_name = 'res' + str(stage) + block

    f1 = out_filters
    f2 = out_filters
    f3 = out_filters
    with tf.variable_scope(block_name):
        X_shortcut = X_input

        #first
        W_conv1 = weight_variable([1, 1, in_filter, f1],trainable=trainable)
        X = tf.nn.conv2d(X_input, W_conv1, strides=[1, 1, 1, 1], padding='SAME')
        X = tf.layers.batch_normalization(X, axis=3, training=training)
        X = tf.nn.relu(X)

        #second
        W_conv2 = weight_variable([kernel_size, kernel_size, f1, f2],trainable=trainable)
        X = tf.nn.conv2d(X, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
        X = tf.layers.batch_normalization(X, axis=3, training=training)
        X = tf.nn.relu(X)

        #third

        W_conv3 = weight_variable([1, 1, f2, f3],trainable=trainable)
        X = tf.nn.conv2d(X, W_conv3, strides=[1, 1, 1, 1], padding='VALID')
        X = tf.layers.batch_normalization(X, axis=3, training=training)

        #final step
        add = tf.add(X, X_shortcut)
        add_result = tf.nn.relu(add)

    return add_result
def convolutional_block( X_input, kernel_size, in_filter,
                        out_filters, stage, block, training, stride=2,trainable = True):
    """
    卷积残差块，shortcut经过一次卷积，主路径3层卷积
    Implementation of the convolutional block as defined in Figure 4
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    kernel_size -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    training -- train or test for bn
    stride -- Integer, specifying the stride to be used
    trainable -- train or test for filter
    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    block_name = 'res' + str(stage) + block
    with tf.variable_scope(block_name):
        f1, f2, f3 = out_filters

        x_shortcut = X_input
        #first
        W_conv1 = weight_variable([1, 1, in_filter, f1],trainable=trainable)
        X = tf.nn.conv2d(X_input, W_conv1,strides=[1, stride, stride, 1],padding='VALID')
        X = tf.layers.batch_normalization(X, axis=3, training=training)
        X = tf.nn.relu(X)

        #second
        W_conv2 = weight_variable([kernel_size, kernel_size, f1, f2],trainable=trainable)
        X = tf.nn.conv2d(X, W_conv2, strides=[1,1,1,1], padding='SAME')
        X = tf.layers.batch_normalization(X, axis=3, training=training)
        X = tf.nn.relu(X)

        #third
        W_conv3 = weight_variable([1,1, f2,f3],trainable=trainable)
        X = tf.nn.conv2d(X, W_conv3, strides=[1, 1, 1,1], padding='VALID')
        X = tf.layers.batch_normalization(X, axis=3, training=training)

        #shortcut path
        W_shortcut = weight_variable([1, 1, in_filter, f3],trainable=trainable)
        x_shortcut = tf.nn.conv2d(x_shortcut, W_shortcut, strides=[1, stride, stride, 1], padding='VALID')

        #final
        add = tf.add(x_shortcut, X)
        add_result = tf.nn.relu(add)

    return add_result
import tensorflow as tf
import numpy as np
import os

def get_file(file_dir,print_map = False):
    '''
    
    :param file_dir: 训练集文件夹根目录
    :param print_map: bool值，是否打印label名称与其转换成的数字对应关系
    :return: 图片列表，label列表
    '''

    images = []#图片
    temp = []#文件夹
    for root, sub_folders, files in os.walk(file_dir):#遍历文件夹
        #root
        #所指的是当前正在遍历的这个文件夹的本身的地址
        #dirs
        # 是一个
        # list ，内容是该文件夹中所有的目录的名字(不包括子目录)
        # files
        # 同样是
        # list, 内容是该文件夹中所有的文件(不包括子目录)
        for name in files:
            images.append(os.path.join(root, name))
        for name in sub_folders:
            temp.append(os.path.join(root, name))
    labels = []
    map = {} #字典 记录分类标签名称与其分类值的对应关系
    map_to = 0 #每个对象映射到的int值
    for one_folder in temp:
        n_img = len(os.listdir(one_folder))
        letter = one_folder.split('/')[-1]

        labels = np.append(labels, n_img * [map_to])
        map[letter] = map_to
        map_to += 1

        # for key,value in map.items():#将增加分类对象数
        #     if letter == key:
        #         labels = np.append(labels,n_img * [value])
        #         break
        # if letter == 'cat':
        #     labels = np.append(labels, n_img * [0])
        # else:
        #     labels = np.append(labels, n_img * [1])
    # shuffle
    if(print_map):
        for key,value in map.items():
            print(str(key) + "  " + str(value))

    temp = np.array([images, labels])
    temp = temp.transpose()
    np.random.shuffle(temp)#打乱
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(float(i)) for i in label_list]
    return image_list,label_list
def get_batch(image_list, label_list, img_width, img_height, batch_size, capacity):
    '''
    
    :param image_list: get_file 的第一个返回值，读取出来的图片列表
    :param label_list: get_file 的第二个返回值，读取出来的label列表
    :param img_width:  模型要求输入的图片宽度
    :param img_height: 模型要求输入的图片高度
    :param batch_size: 一个块的大小
    :param capacity:   数据池的最大容量
    :return: 标准化后的图片数据块，label块
    '''
    image = tf.cast(image_list, tf.string)#将图片转化为string
    label = tf.cast(label_list, tf.int32)#将结果转化为int

    input_queue = tf.train.slice_input_producer([image, label])#从 本地文件里抽取tensor，准备放入Filename Queue（文件名队列）中

    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)#解码

    image = tf.image.resize_image_with_crop_or_pad(image, img_width, img_height)
    image = tf.image.per_image_standardization(image)  # 将图片标准化
    image_batch, label_batch = tf.train.batch([image, label], batch_size=batch_size, num_threads=64, capacity=capacity)#从文件名队列中提取tensor，使用单个或多个线程，准备放入文件队列，enqueuemany = False,输出[batch_size,image,label]
    label_batch = tf.reshape(label_batch, [batch_size])

    return image_batch, label_batch
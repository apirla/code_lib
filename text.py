import tensorflow as tf
from tensorflow_lib.models import weight_variable

x = weight_variable([20],True,64,2)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

print(sess.run(x.value()))
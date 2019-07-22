import tensorflow as tf
from tensorflow_lib.models import weight_variable
from tensorflow_lib.input import get_file

img,label = get_file('D:/python文件/code_lib/tensorflow_lib/train',print_map=True)

print(img)
print(label)
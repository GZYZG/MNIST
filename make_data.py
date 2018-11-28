'''
本模块用于制作类似MNIST数据集格式的数据
参考自：https://blog.csdn.net/ch15717502064/article/details/78006715
'''

from PIL import Image
import struct
import os

data_file_path = 'D:\\Study\\ML&CV\\TensorFlow\\Flow\\my_data\\'


img_file_path = 'D:\\Study\\ML&CV\\video_analysis\\data_set\\handwrite\\small_imgs\\'
img_num = 270
label_num = 270
rows = 200
cols = 200

imgUbyte = data_file_path+'train-image-idx3-ubyte'
labelUbyte = data_file_path+'train-label-idx1-ubyte'
label = 'D:\\Study\\ML&CV\\video_analysis\\data_set\\handwrite\\small_imgs\\label.txt'

def readImage():
    file_ubyte = open(labelUbyte,'wb')
    file_txt = (label, 'r')

    file_ubyte.write(struct.pack('i', 17301504))
    file_ubyte.write(struct.pack('i', img_num))
    file_ubyte.write(struct.pack('i',rows))
    file_ubyte.write(struct.pack('i', cols))

    for i in range(img_num):
        img = Image.open(img_file_path)

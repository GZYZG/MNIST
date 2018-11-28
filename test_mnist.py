import input_data
import tensorflow as tf
import cv2


def imageprepare(image):
    l = image.reshape(-1,28*28)[0]
    tva = [x*1.0/255.0 for x in l]
    return tva

#MNIST数据输入
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

print('load data successfully')

#图像输入向量
x = tf.placeholder(tf.float32, [None, 784])
#权重
W = tf.Variable(tf.zeros([784, 10]))
#偏置
b = tf.Variable(tf.zeros([10]))

#进行模型的计算， y是预测值，_y是实际值
y = tf.nn.softmax(tf.matmul(x,W) + b)
_y = tf.placeholder('float', [None, 10])

#计算交叉熵
cross_entropy = -tf.reduce_sum(_y * tf.log(y))
#使用BP算法进行微调， 以0.01的学习率进行学习
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

#添加初始化变量的操作
init = tf.initialize_all_variables()

#启动创建的模型，并初始化变量
sess = tf.Session()
sess.run(init)

#开始训练模型，循环训练一千次
for i in range(10000):
    #随机抓取训练数据中的100个数据进行训练
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x:batch_xs, _y:batch_ys})
    if i % 500 == 0:
        correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(_y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

        print('has trained for ', i, ' times ', 'accuracy:', sess.run(accuracy, feed_dict={x:mnist.test.images, _y:mnist.test.labels}))

#进行模型评估
#判断预测标签和实际标签是否匹配
correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(_y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

print(sess.run(accuracy, feed_dict={x:mnist.test.images, _y:mnist.test.labels}))









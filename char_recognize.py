import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
 
# 载入数据集
mnist = input_data.read_data_sets("MNIST_data",one_hot=True)
 
# 批次的大小
batch_size = 100
 
# 计算一共有多少批次
n_batch = mnist.train.num_examples // batch_size
 
# 定义两个placeholder
x = tf.placeholder(tf.float32,[None,28*28])
y = tf.placeholder(tf.float32,[None,10])
# 训练的时候只是修改一部分神经元
keep_prob = tf.placeholder(tf.float32)
# 定义初始的学习率
lr = tf.Variable(0.001, dtype=tf.float32)
 
# 创建一个简单的神经网络
# 第二层有2000个神经元
W1 = tf.Variable(tf.truncated_normal([28*28, 500],stddev=0.1))    # 使用正态分布初始化
b1 = tf.Variable(tf.zeros([500])+0.1)
L1 = tf.nn.tanh(tf.matmul(x,W1)+b1)
# 有百分之多少的神经元在工作
L1_drop = tf.nn.dropout(L1,keep_prob)
 
W2 = tf.Variable(tf.truncated_normal([500,300],stddev=0.1))    # 使用正态分布初始化
b2 = tf.Variable(tf.zeros([300])+0.1)
L2 = tf.nn.tanh(tf.matmul(L1_drop,W2)+b2)
# 有百分之多少的神经元在工作
L2_drop = tf.nn.dropout(L2,keep_prob)
 
# 输出层
W3 = tf.Variable(tf.truncated_normal([300,10],stddev=0.1))    # 使用正态分布初始化
b3 = tf.Variable(tf.zeros([10])+0.1)
prediction = tf.nn.softmax(tf.matmul(L2_drop,W3)+b3)
 
# 二次代价函数
# loss = tf.reduce_mean(tf.square(y-prediction))
# 对数似然代价函数
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
# 使用梯度下降法
train_step = tf.train.AdamOptimizer(lr).minimize(loss)
 
# 初始化变量
init = tf.global_variables_initializer()
 
# 计算正确率,下面是一个布尔型列表
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
# 求准确率，首先把布尔类型转化为浮点类型
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
 
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(51):
        # 指数变化学习率
        sess.run(tf.assign(lr, 0.001 * (0.95**epoch)))
        for batch in range(n_batch):
            # 使用函数获取一个批次图片
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x:batch_xs,y:batch_ys,keep_prob:1.0})
 
        acc = sess.run(accuracy,feed_dict={x:mnist.test.images, y:mnist.test.labels,keep_prob:1.0})
        print("Iter " + str(epoch) + ",Testing Accuracy " + str(acc))

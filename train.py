import time
from datetime import timedelta
import math
import numpy as np
import tensorflow as tf
import random
import dataset
from tensorflow import set_random_seed
from numpy.random import seed
seed(10)  # 确定一个随机数的种子
set_random_seed(20)

batch_size = 32  # 指定32张图片

classes = ['dogs','cats']  # 指定类别标签
num_classes = len(classes)

validation_size = 0.2  # 设定0.2为测试比例
img_size = 64  # 让所有的输入图像的大小都一样,全连接层 64*64
num_channels = 3  # 颜色通道
train_path = "training_data"  # 指定训练路径
data = dataset.read_train_sets(train_path, img_size, classes, validation_size)

# 开始完成神网络的架构
session = tf.Session()
x = tf.placeholder(tf.float32,[None, img_size,img_size,num_channels],name='x')  # 指定输入的x
y_true = tf.placeholder(tf.float32,[None,num_classes],name='y_true')  # 指定y的检验样本
y_true_cls = tf.argmax(y_true, dimension=1,name='y_true_cls')  # 按照行来找出最大的索引

# 网络参数
# 卷积网络参数
filter_size_conv1 = 3
num_filter_conv1 = 32

filter_size_conv2 = 3
num_filter_conv2 = 32

filter_size_conv3 = 3
num_filter_conv3 = 64
# 全连接层的参数
fc_layer_size = 1024

# 创造权重矩阵


def create_weight(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

# 创造偏置


def create_biases(size):
    return  tf.Variable(tf.constant(0.05, shape=[size]))

def create_con_layer(input, num_input_channels, conv_filter_size,num_filters):
    weight = create_weight(shape=[conv_filter_size,conv_filter_size,num_input_channels,num_filters])
    biases = create_biases(num_filters)
    layer = tf.nn.conv2d(input,weight,strides=[1,1,1,1],padding="SAME")
    layer += biases
    # 激励函数
    layer = tf.nn.relu(layer)
    # 池化
    layer = tf.nn.max_pool(value=layer,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
    return layer


def create_flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()  # Returns the total number of elements, or none for incomplete shapes
    layer = tf.reshape(layer,[-1,num_features])
    return layer

def create_fc_layer(input,num_inputs,num_outputs,use_relu = True):
    weight = create_weight(shape=[num_inputs,num_outputs])
    biases = create_biases(num_outputs)
    layer = tf.matmul(input,weight) + biases
    layer = tf.nn.dropout(layer,keep_prob=0.7)  # 防止神经网络过拟合，保留0.7
    if use_relu:
        layer = tf.nn.relu(layer)
    return layer


def show_progress(epoch,feed_train_dict,feed_valid_dict,val_loss,i):
    acc = session.run(accuracy,feed_dict=feed_train_dict)
    val_acc = session.run(accuracy,feed_dict=feed_valid_dict)
    print("训练到了第{}个epoch,迭代了{}次,Training Accruracy {},Validation Accruacy{},损失{}".format(epoch+1,i,acc,val_acc,val_loss))



layer_conv1 = create_con_layer(input=x,
                                num_input_channels=num_channels,
                                conv_filter_size=filter_size_conv1,
                                num_filters=num_filter_conv1)
layer_conv2 = create_con_layer(input=layer_conv1,
                               num_input_channels=num_filter_conv1,
                               conv_filter_size=filter_size_conv2,
                               num_filters=num_filter_conv2)
layer_conv3 = create_con_layer(input=layer_conv2,
                               num_input_channels=num_filter_conv2,
                               conv_filter_size=filter_size_conv3,
                               num_filters=num_filter_conv3)
layer_flat = create_flatten_layer(layer_conv3)
layer_fc1 = create_fc_layer(layer_flat,
                            num_inputs=layer_flat.get_shape()[1:4].num_elements(),
                            num_outputs=fc_layer_size)
layer_fc2 = create_fc_layer(layer_fc1,
                            num_inputs=fc_layer_size,
                            num_outputs=num_classes,
                            use_relu=False)
y_pred = tf.nn.softmax(layer_fc2,name='y_pred')
y_pred_cls = tf.argmax(y_pred, dimension=1)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = y_pred, labels=y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(1e-5).minimize(cost)
correct_prediction = tf.equal(y_pred_cls,y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

session.run(tf.global_variables_initializer())
total_iterations = 0
saver = tf.train.Saver()  # 保存模型,读入模型
def train(num_iteration):
    global total_iterations
    for i in range(total_iterations,total_iterations + num_iteration):
        x_batch,y_true_batch,_,cls_batch = data.train.next_batch(batch_size)
        x_valid_batch, y_valid_batch,_, valid_cls_batch = data.valid.next_batch(batch_size)
        session.run(optimizer,feed_dict={x:x_batch,y_true:y_true_batch})
        if i % int(data.train.num_examples/batch_size) == 0:
            val_loss = session.run(cost,feed_dict={x:x_batch,y_true:y_true_batch})
            epoch = int(i/(data.train.num_examples/batch_size))
            show_progress(epoch,{x:x_batch,y_true:y_true_batch},{x:x_valid_batch,y_true:y_valid_batch},val_loss,i)
            saver.save(session,'train_model3/dog-cat.ckpt',global_step=i)  # global_step 第i次模型迭代保存

    total_iterations += num_iteration
train(8000)
import tensorflow as tf
import numpy as np
import os,glob,cv2

img_size = 64
num_channels = 3
images = []
path = 'dog.1081.jpg'
image = cv2.imread(path)
image = cv2.resize(image,(img_size,img_size),0,0,cv2.INTER_LINEAR)
images.append(image)
images = np.array(images,dtype=np.float32)
images = np.multiply(images,1.0/255.0)

x_batch = images.reshape(-1,img_size,img_size,num_channels)
session = tf.Session()
saver = tf.train.import_meta_graph('train_model3/dog-cat.ckpt-7925.meta')  # 加载网络结构图
saver.restore(session,'train_model3/dog-cat.ckpt-7925')  # 加载权重参数
graph = tf.get_default_graph()  # 得到加载的图
y_pred = graph.get_tensor_by_name("y_pred:0") # 形如'conv1'是节点名称，而'conv1:0'是张量名称，表示节点的第一个输出张量
x= graph.get_tensor_by_name("x:0")
feed_dict_testing = {x: x_batch}
result=session.run(y_pred, feed_dict=feed_dict_testing)
res_label = ['dog','cat']
print(res_label[result.argmax()])


# 没有命名导致的错误

'''
For graph.get_tensor_by_name("prediction:0") to work you should have named it when you created it. This is how you can name it

prediction = tf.nn.softmax(tf.matmul(last,weight)+bias, name="prediction")
If you have already trained the model and can't rename the tensor, you can still get that tensor by its default name as in,

y_pred = graph.get_tensor_by_name("Reshape_1:0")
If Reshape_1 is not the actual name of the tensor, you'll have to look at the names in the graph and figure it out. You can inspect that with

for op in graph.get_operations():
    print(op.name)
'''

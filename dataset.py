import glob
import os
import cv2
import numpy as np
from sklearn.utils import shuffle

class DataSet(object):
    def __init__(self,images,labels,img_names,cls):
        self._num_examples = images.shape[0]  # 就是batchsize
        self._images = images
        self._labels = labels
        self._img_names = img_names
        self._cls = cls
        self._epochs_done = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def img_names(self):
        return self._img_names

    @property
    def cls(self):
        return self._cls

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_done(self):
        return self._epochs_done

    def next_batch(self,batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            self._epochs_done += 1
            self._index_in_epoch = 0
            assert  batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end], self._img_names[start:end], self._cls[start:end]


class DataSets(object):
    pass


def read_train_sets(train_path, image_size, classes, validation_size):
    images, labels, image_names, cls = load_train(train_path,image_size,classes)
    images, labels, image_names, cls = shuffle(images, labels, image_names, cls)  # 这是一种整体打乱,让计算机一会看狗一会看猫交替进行
    data_sets = DataSets()
    if isinstance(validation_size,float):
        validation_size = int(validation_size * images.shape[0]) # 指定好验证集的个数
    validation_images = images[:validation_size]
    validation_labels = labels[:validation_size]
    validation_image_name = image_names[:validation_size]
    validation_cls = cls[:validation_size]  # 获得训练集

    train_images = images[validation_size:]
    train_labels = labels[validation_size:]
    train_image_name = image_names[validation_size:]
    train_cls = cls[validation_size:]

    data_sets.train = DataSet(train_images,train_labels,train_image_name,train_cls)
    data_sets.valid = DataSet(validation_images, validation_labels, validation_image_name, validation_cls)  # 获取验证集

    return data_sets


def load_train(train_path, image_size, classes):
    images = []
    labels = []
    img_names = []
    cls = []
    print("开始取得数据...")
    for fields in classes:
        index = classes.index(fields)
        print("现在读{},的Index{}".format(fields,index))
        path = os.path.join(train_path,fields,'*g') # 取得所有以g结尾的文件
        files = glob.glob(path) # Return a list of paths matching a pathname pattern
        for file_path in files:
            image = cv2.imread(file_path)
            image = cv2.resize(image,(image_size,image_size),interpolation= cv2.INTER_LINEAR)  # 这里不用指定fx = (double)dsize.width/src.cols,双线性插值
            image = image.astype(np.float32)
            image = np.multiply(image,1.0/255.0)  # 归一化处理
            images.append(image)
            label = np.zeros(len(classes))
            label[index] = 1.0  # 做一个one_hot数据标签
            labels.append(label)
            file_base = os.path.basename(file_path)  #Returns the final component of a pathname
            img_names.append(file_base)
            cls.append(fields)
    images = np.array(images)
    labels = np.array(labels)
    cls = np.array(cls)

    return images, labels, img_names, cls
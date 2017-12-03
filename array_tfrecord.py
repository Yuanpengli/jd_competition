#!/usr/bin/env python
# encoding: utf-8
# Created Time:

import os
import numpy as np
import tensorflow as tf


def convert_string_to_dtype_np(s):
    if s in ['uint8']:
        return np.uint8

    elif s in ['int32', 'np.int32']:
        return np.int32

    elif s in ['int64', 'np.int64']:
        return np.int64

    elif s in ['float32', 'np.float32']:
        return np.float32

    elif s in ['float64', 'np.float64']:
        return np.float64

    else:
        raise ValueError("Invalid string: {0}.".format(s))


def convert_string_to_dtype_tf(s):
    import tensorflow as tf
    if s in ['uint8']:
        return tf.uint8

    elif s in ['int32', 'tf.int32']:
        return tf.int32

    elif s in ['int64', 'tf.int64']:
        return tf.int64

    elif s in ['float32', 'tf.float32']:
        return tf.float32

    elif s in ['float64', 'tf.float64']:
        return tf.float64

    else:
        raise ValueError("Invalid string: {0}.".format(s))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def to_list_files(tfrecord_files):
    if not isinstance(tfrecord_files, list):
        tfrecord_files = [tfrecord_files]

    for name in tfrecord_files:
        if not os.path.exists(name):
            raise ValueError('file {0} does not exists'.format(name)) 

    return tfrecord_files


class BaseArrayTFRecord(object):

    def write_tfrecord(self, tfrecord_file, x_list, y_list, verbose=False, log_num=1000):
        writer = tf.python_io.TFRecordWriter(tfrecord_file)
        ii = 0
        for x, y in zip(x_list, y_list):
            # x_array, y_array = self.to_array(x, y)
            self.write_example(writer, x, y)
            if verbose and ii == 10:
                print('Finished {0}'.format(ii))
                ii += 1
        writer.close()

    def read_tfrecord(self, tfrecord_files, num_epochs=1):
        tfrecord_files = to_list_files(tfrecord_files)
        filename_queue = tf.train.string_input_producer(tfrecord_files, num_epochs=num_epochs)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        single_example = self.read_example(serialized_example)

        return single_example

    def reconstruct_tfrecord(self, tfrecord_files, num_examples=None):
        tfrecord_files = to_list_files(tfrecord_files)
        tfrecord_iterator = tf.python_io.tf_record_iterator(path=tfrecord_files[0])
        reconstructed_examples = []
        ii = 0
        for record_string in tfrecord_iterator:
            example = self.reconstruct_example(record_string)
            reconstructed_examples.append(example)
            ii += 1
            if ii == num_examples:
                break

        return reconstructed_examples

    def to_array(self, x, y):
        return x, y

    def write_example(self, writer, x, y):
        """
        case 1: (both x and y are arrays)
            x_raw = x.tostring()
            y_raw = y.tostring()
            example = tf.train.Example(features=tf.train.Features(
                feature={
                    'x_raw': _bytes_feature(x_raw),
                    'y_raw': _bytes_feature(y_raw)})

            writer.write(example.SerializeToString())

        case 2: (x is array, y is int)
            x_raw = x.tostring()
        """
        raise NotImplementedError()

    def read_example(self, serialized_example):
        """
        features = tf.parse_single_example(
                serialized_example,
                features={
                    'name_int_1': tf.FixedLenFeature([], tf.int64),
                    'name_byte_1': tf.FixedLenFeature([], tf.string),
                    ...})

        array_1 = tf.decode_raw(features['name_byte_1'], tf.unit8)
        label_1 = tf.cast(features['name_int_1'], tf.int32)

        return array_1, label_1
        """
        raise NotImplementedError()

    def reconstruct_example(self, tfrecord_string):
        """
        """
        raise NotImplementedError()


class ArrayTFRecord(BaseArrayTFRecord):
    def __init__(self, image_shape, labels_shape, image_dtype, labels_dtype):
        self.image_shape = image_shape
        self.labels_shape = labels_shape
        self.image_dtype = image_dtype
        self.labels_dtype = labels_dtype

    def write_example(self, writer, x, y):
        assert x.shape == self.image_shape
        assert y.shape == self.labels_shape
        assert str(x.dtype) == self.image_dtype
        assert str(y.dtype) == self.labels_dtype

        x_feature = x.tostring()
        y_feature = y.tostring()
        feature = {'x': _bytes_feature(x_feature), 
                   'y': _bytes_feature(y_feature)}

        example = tf.train.Example(features=tf.train.Features(
            feature=feature))
        writer.write(example.SerializeToString())

    def read_example(self, serialized_example):
        # allowed dtype tf.string, tf.float32, tf.int64
        features = tf.parse_single_example(
                serialized_example,
                features={
                    'img_raw': tf.FixedLenFeature([], tf.string), 
                    'img_label': tf.FixedLenFeature([], tf.string)})

        x = tf.decode_raw(features['img_raw'], convert_string_to_dtype_tf(self.image_dtype))
        y = tf.decode_raw(features['img_label'], convert_string_to_dtype_tf(self.labels_dtype))

        x = tf.reshape(x, self.image_shape, name='image')
        y = tf.reshape(y, self.labels_shape, name='label')
        return x, y

    def reconstruct_example(self, tfrecord_string):
        example = tf.train.Example()
        example.ParseFromString(tfrecord_string)
        feature = example.features.feature
        x = feature['x'].bytes_list.value[0]
        x = np.fromstring(x, dtype=convert_string_to_dtype_np(self.image_dtype))
        x = np.reshape(x, self.image_shape)

        y = feature['y'].bytes_list.value[0]
        y = np.fromstring(y, dtype=convert_string_to_dtype_np(self.labels_dtype))
        y = np.reshape(y, self.labels_shape)

        return x, y


def preprocess_example(single_example):
    single_example = list(single_example)
    # print single_example
    single_example[0] = tf.cast(single_example[0], tf.float32)

    single_example[1] = tf.reshape(tf.squeeze(single_example[1]), [-1, 30])
    # single_example[1] = tf.to_int32(single_example[1])
    #single_example[1] = tf.one_hot(single_example[1], depth=3, on_value=1.0, off_value=0.0)
    #single_example[1] = tf.squeeze(single_example[1])
    return single_example


def shuffle_batch_example(single_example, batch_size):
    if not isinstance(single_example, (tuple, list)):
        single_example = [single_example]

    batch_example = tf.train.shuffle_batch(
            list(single_example),
            batch_size=batch_size,
            num_threads=2,
            capacity=100 * 3 * batch_size,
            min_after_dequeue=100)

    return batch_example


def prepare_batch_data(train_tfrecord_file, valid_tfrecord_file, batch_size, num_epochs=1, graph=None):
    if not isinstance(graph, tf.Graph):
        graph = tf.Graph()

    with graph.as_default():
        array_tfrecord = ArrayTFRecord(image_shape, labels_shape, image_dtype, labels_dtype)

        train_single_example = array_tfrecord.read_tfrecord(train_tfrecord_file, num_epochs=num_epochs)
        train_single_example = preprocess_example(train_single_example)
        train_batch_example = shuffle_batch_example(train_single_example, batch_size)
        valid_single_example = array_tfrecord.read_tfrecord(valid_tfrecord_file, num_epochs=num_epochs)
        valid_single_example = preprocess_example(valid_single_example)
        valid_batch_example = shuffle_batch_example(valid_single_example, batch_size)
        return train_batch_example, valid_batch_example


image_shape = (1280, 720, 3)
labels_shape = (30, 1)
image_dtype = 'uint8'
labels_dtype = 'float64'


batch_size = 3
num_epochs = 10
'''
valid_list = [9 * i for i in range(31)]
train_list = [i for i in range(296) if i not in valid_list]
train_tfrecord_file = ['/home/yunzhou/fromFiles/datasets/jd_data/temp/jd_img_' + str(num) + '.tfrecord' for num in train_list]
valid_tfrecord_file = ['/home/yunzhou/fromFiles/datasets/jd_data/temp/jd_img_' + str(num) + '.tfrecord' for num in valid_list]
'''

import os
dataset_dir = '/home/yunzhou/fromFiles/datasets/jd_data/temp/'
name_list = os.listdir(dataset_dir)
tfrecord_file = [os.path.join(dataset_dir, name) for name in name_list]
train_tfrecord_file = tfrecord_file[:270]
valid_tfrecord_file = tfrecord_file[270:]

train_batch_example, valid_batch_example = prepare_batch_data(train_tfrecord_file, valid_tfrecord_file, batch_size, num_epochs)
train_batch_images, train_batch_labels = train_batch_example
valid_batch_images, valid_batch_labels = valid_batch_example

print (train_batch_images)
print (train_batch_labels)





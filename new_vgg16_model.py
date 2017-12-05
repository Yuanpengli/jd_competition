import tensorflow as tf
from keras.models import load_model, Model, Sequential
from keras.layers import Input, Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization as BNOR
from keras.regularizers import l2

class Model(object):
    def __init__(self, num_class=30):
        self.num_class = num_class

    def _vgg16(self, input_tensor):
        h = Conv2D(64, (3, 3), activation='relu', strides=1, padding='same', name='block1_conv1')(input_tensor)
        h = Conv2D(64, (3, 3), activation='relu', strides=1, padding='same', name='block1_conv2')(h)
        h = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block1_pool')(h)
        print ('block 1')
        print ('h', h)
        h = Conv2D(128, (3, 3), activation='relu', strides=1, padding='same', name='block2_conv1')(h)
        h = Conv2D(128, (3, 3), activation='relu', strides=1, padding='same', name='block2_conv2')(h)
        h = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block2_pool')(h)
        print ('block 2')
        print ('h', h)
        h = Conv2D(256, (3, 3), activation='relu', strides=1, padding='same', name='block3_conv1')(h)
        h = Conv2D(256, (3, 3), activation='relu', strides=1, padding='same', name='block3_conv2')(h)
        h = Conv2D(256, (3, 3), activation='relu', strides=1, padding='same', name='block3_conv3')(h)
        h = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block3_pool')(h)
        print ('block 3')
        print ('h', h)
        h = Conv2D(512, (3, 3), activation='relu', strides=1, padding='same', name='block4_conv1')(h)
        h = Conv2D(512, (3, 3), activation='relu', strides=1, padding='same', name='block4_conv2')(h)
        h = Conv2D(512, (3, 3), activation='relu', strides=1, padding='same', name='block4_conv3')(h)
        h = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block4_pool')(h)
        print ('block 4')
        print ('h', h)
        h = Conv2D(512, (3, 3), activation='relu', strides=1, padding='same', name='block5_conv1')(h)
        h = Conv2D(512, (3, 3), activation='relu', strides=1, padding='same', name='block5_conv2')(h)
        h = Conv2D(512, (3, 3), activation='relu', strides=1, padding='same', name='block5_conv3')(h)
        h = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block5_pool')(h)
        print ('block 5')
        print ('h', h)
        # model.load_weights('~/.keras/models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
        return h

    def inference(self, input_tensor):
        vgg_output = self._vgg16(input_tensor)
        output = Flatten(name='flatten_1')(vgg_output)
        h = Dense(200, activation='relu')(output)
        predict = Dense(self.num_class, activation='relu')(h)
        return predict

    def loss_func(self, logits, true_labels):
        loss = tf.reduce_mean((tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                       labels=true_labels,
                                                                       name="entropy")))
        self.loss = loss
        return loss

    def minimize(self, loss):
        train_op = tf.train.AdamOptimizer(0.0001).minimize(loss)
        return train_op




'''
dataset_dir = '/home/liyuanpeng/Documents/minist'
loader = MnistLoader()
loader.load_dataset(dataset_dir)
dataset = Dataset(loader=loader)

input_x = dataset.data[0]
input_x = input_x.reshape([-1, 28, 28, 1])
label = dataset.labels[0]

x = Input(shape=(28, 28, 1))
output = vgg16(x)
# output = tf.reshape(output, [-1, 512])
output = Flatten(name='flatten_1')(output)
print (output)
h = Dense(100, activation='relu')(output)
predict = Dense(10, activation='softmax')(h)
model = Model(x, predict)
model.compile(optimizer='Adam',
          loss='sparse_categorical_crossentropy',
          metrics=['accuracy'])
model.fit(input_x, label, validation_split=0.1, batch_size=128)
'''

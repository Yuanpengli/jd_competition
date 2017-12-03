import tensorflow as tf
from keras.models import load_model, Model, Sequential
from keras.layers import Input, Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization as BNOR
from keras.regularizers import l2

from Teemo.datasets.loaders.mnist_loader import MnistLoader
from Teemo.datasets.base import Dataset

def vgg16(input_placeholder):
    h = Conv2D(64, (3, 3), activation='relu', strides=1, padding='same', name='block1_conv1')(input_placeholder)
    h = Conv2D(64, (3, 3), activation='relu', strides=1, padding='same', name='block1_conv2')(h)
    h = MaxPooling2D( pool_size=(2,2),strides=(2,2),name='block1_pool' )(h)
    print ('block 1')
# block 2
    h = Conv2D(128, (3, 3), activation='relu', strides=1, padding='same', name='block2_conv1')(h)
    h = Conv2D(128, (3, 3), activation='relu', strides=1, padding='same', name='block2_conv2')(h)
    h = MaxPooling2D( pool_size=(2,2),strides=(2,2),name='block2_pool')(h)
# block 3
    h = Conv2D(256, (3, 3), activation='relu', strides=1, padding='same', name='block3_conv1')(h)
    h = Conv2D(256, (3, 3), activation='relu', strides=1, padding='same', name='block3_conv2')(h)
    h = Conv2D(256, (3, 3), activation='relu', strides=1, padding='same', name='block3_conv3')(h)
    h = MaxPooling2D( pool_size=(2,2),strides=(2,2),name='block3_pool')(h)
# block 4
    h = Conv2D(512, (3, 3), activation='relu', strides=1, padding='same', name='block4_conv1')(h)
    h = Conv2D(512, (3, 3), activation='relu', strides=1, padding='same', name='block4_conv2')(h)
    h = Conv2D(512, (3, 3), activation='relu', strides=1, padding='same', name='block4_conv3')(h)
    h = MaxPooling2D( pool_size=(2,2),strides=(2,2),name='block4_pool' )(h)
    print ('h', h)
#block 5
    h = Conv2D(512, (3, 3), activation='relu', strides=1, padding='same', name='block5_conv1')(h)
    h = Conv2D(512, (3, 3), activation='relu', strides=1, padding='same', name='block5_conv2')(h)
    h = Conv2D(512, (3, 3), activation='relu', strides=1, padding='same', name='block5_conv3')(h)
    print ('block 5')
    print ('h', h)
    h = MaxPooling2D( pool_size=(2,2),strides=(2,2),name='block5_pool' )(h)
	#model.load_weights('~/.keras/models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
    return h

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
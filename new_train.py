import os
import tensorflow as tf
import numpy as np

from foundation.datasets.tf import ImageFilesDataset
#from foundation.dataset
from jd_competition.new_vgg16_model import Model


if __name__=="__main__":
    dataset_dir = '/home/liyuanpeng/Desktop/jingdong_competition/train_imgs/1'
    checkpoint_dir = ''

    name_list = os.listdir(dataset_dir)
    target_size, num_class, batch_size = (1280, 720), 30, 8
    image_files = [os.path.join(dataset_dir, name) for name in name_list]
    image_labels = [0]*2950
    dataset_img = ImageFilesDataset(target_size, batch_size=batch_size)
    # valid_ops = dataset_img.compile(decoding=True, grayscale=False,
    #                             one_hot=True, num_class=num_class)
    train_ops = dataset_img.compile(decoding=True, grayscale=False,
                                    one_hot=True, num_class=num_class,
                                    distortion=True, shuffle=True, buffer_size=300,
                                    epochs=10)
    ops = train_ops

    sess = tf.Session()
    feed_dict = {dataset_img.image_files: image_files,
                 dataset_img.image_labels: image_labels}

    sess.run(ops['dataset_init_op'], feed_dict=feed_dict)

    next_element = ops['next_element']
    image, label = next_element

    model = Model(checkpoint_dir=checkpoint_dir)
    logits = model.inference(image)
    loss = model.loss_func(logits, label)
    train_op = model.minimize(loss)

    for ii in range(10000):
        _, loss_value = sess.run(train_op, loss)
        print (ii, loss_value)
        if ii % 100 == 0:
            model.save(sess, global_step=ii)

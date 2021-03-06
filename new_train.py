import os
import tensorflow as tf
import numpy as np

from foundation.datasets.tf import ImageFilesDataset
#from foundation.dataset
from jd_competition.new_vgg16_model import Model

def save_file(dir, dir_int):

    file_path_list = os.listdir(dir)
    file_path_list = [os.path.join(dir, file_ii) for file_ii in file_path_list]
    num_file = len(file_path_list)
    label_list = [dir_int]*num_file
    return file_path_list, label_list

def comb_all_file(img_dir):

    file_dir = os.listdir(img_dir)
    label_list = [int(name) for name in file_dir]


    all_label = []
    all_image_path_list = []

    for file_path, dir_int in zip(file_dir, label_list):

        pig_dir = os.path.join(img_dir, file_path)
        file_path_list, label_list = save_file(pig_dir, dir_int)
        all_image_path_list += file_path_list
        all_label += label_list
    return all_image_path_list, all_label

def order_file_ten(img_dir):
    file_dir = os.listdir(img_dir)
    label_list = [int(name) for name in file_dir]

    all_label = []
    all_image_path_list = []

    for file_path, dir_int in zip(file_dir, label_list):
        pig_dir = os.path.join(img_dir, file_path)
        file_path_list, label_list = save_file(pig_dir, dir_int)
        all_image_path_list.append(file_path_list)
        all_label.append(label_list)

    all_image_list = []
    all_label_list = []
    ii = 0
    while True:

        if ii ==300:
            break
        for jj in range(30):
            all_image_list += all_image_path_list[jj][ii*10:(ii+1)*10]
            all_label_list += all_label[jj][ii*10:(ii+1)*10]
        ii += 1
    return all_image_list, all_label_list   


if __name__=="__main__":
    #dataset_dir = '/home/liyuanpeng/Desktop/jingdong_competition/train_imgs'

    dataset_dir = '/home/yuanpeng/data/train_imgs'

    all_image_path_list, all_label = order_file_ten(dataset_dir)
    image_files = all_image_path_list
    image_labels = all_label
    print (all_image_path_list[:20])
    print (all_label[:20])
    assert len(all_image_path_list) == len(image_labels)

    
    checkpoint_dir = '/home/yuanpeng/save/pigface/'

    name_list = os.listdir(dataset_dir)
    target_size, num_class, batch_size = (1280, 720), 30, 5

    dataset_img = ImageFilesDataset(target_size, batch_size=batch_size)
    # valid_ops = dataset_img.compile(decoding=True, grayscale=False,
    #                             one_hot=True, num_class=num_class)
    train_ops = dataset_img.compile(decoding=True, grayscale=False,
                                    one_hot=True, num_class=num_class,
                                    distortion=True, shuffle=True, buffer_size=900,
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

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    #model.reload(sess)
    for ii in range(100000):
        _, loss_value = sess.run([train_op, loss])
        print ('*'*100)
        print (ii, loss_value)
        if ii % 100 == 0:
            model.save(sess, global_step=ii)
    

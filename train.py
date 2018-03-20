import numpy as np
import tensorflow as tf
from keras.layers import Flatten, Dense
from jd_competition.array_tfrecord import prepare_batch_data
from jd_competition.vgg16 import vgg16
def run_eval(sess, eval_ops, num_batches_per_epoch):
   eval_metrics_tmp = []
   eval_metrics = {}
   count = 0
   while True:
       try:
           # print('eval_ops', eval_ops)
           _metrics = sess.run(eval_ops)
           print ('*'*100)
           print ('_metrics', _metrics)
           # print('metrics', _metrics)
           eval_metrics_tmp.append(_metrics)
           count += 1
           if count == num_batches_per_epoch:
               break
       except tf.errors.OutOfRangeError:
           break
   # print eval_metrics_tmp
   print ('*'*100)
   print (eval_metrics_tmp)
   keys = list(eval_metrics_tmp[0].keys())
   try:
       keys.remove('train_op')
   except ValueError:
       pass

   for key in keys:
       eval_metrics[key] = np.mean([x[key] for x in eval_metrics_tmp])

   # print('num_batches_eval: {0}'.format(count))
   return eval_metrics

def evaluate(logits, label_tensor):
    # print logits
    # print label_tensor
    loss = tf.reduce_mean((tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                          labels=label_tensor,
                                                                          name="entropy")))
    return loss



def inference(image):
    vgg_output = vgg16(image)
    print ('vgg_output:', vgg_output)
    vgg_flatten = Flatten(name='flatten')(vgg_output)
    h = Dense(100, activation='relu')(vgg_flatten)
    predict = Dense(30, activation='relu')(h)
    return predict

def train(train_dataset_path, valid_dataset_path, batch_size, num_epochs):
    with tf.Graph().as_default() as graph:
        train_batch_example, valid_batch_example = prepare_batch_data(train_dataset_path, valid_dataset_path, batch_size, num_epochs, graph)
        train_batch_images, train_batch_labels = train_batch_example
        valid_batch_images, valid_batch_labels = valid_batch_example
        print ('*'*100)
        print ('train_batch_images', train_batch_images)
        with tf.variable_scope('vgg16') as scope:
            train_logits = inference(train_batch_images)
            print ('train_logits',train_logits)
            scope.reuse_variables()
            valid_logits = inference(valid_batch_images)
            print ('valid_logits', valid_logits)
        valid_loss = evaluate(valid_logits, valid_batch_labels)
        train_loss = evaluate(train_logits, train_batch_labels)
        print ('*'*100)
        print ('train_loss', train_loss)
        train_op = tf.train.AdamOptimizer(0.0001).minimize(train_loss)
        saver = tf.train.Saver()



    with tf.Session(graph=graph) as sess:
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        coord = tf.train.Coordinator()

        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # train_metrics = run_eval(sess, {'train_op': train_op, 'loss': total_loss}, 2)
        # eval_metrics = run_eval(sess, {'eval_op': eval_ops}, 1)
        metrics_1 = run_eval(sess, {'train_op': train_op, 'train_loss': train_loss}, 2000)
        # metrics_1 = run_eval(sess, [train_op, train_loss], 2000)
        print(metrics_1)
        metrics_2 = run_eval(sess, {'valid_loss': valid_loss}, 100)
        print(metrics_2)
        print('---' * 10 + str(ii) + '---' * 10)
        saver.save(sess, "/Users/wangxiaodong/LITS_FCN/model.ckpt", ii)
        coord.request_stop()
        coord.join(threads)
        sess.close()

if __name__=="__main__":
    import os
    dataset_dir = '/home/yunzhou/fromFiles/datasets/jd_data/temp/'
    name_list = os.listdir(dataset_dir)
    tfrecord_file = [os.path.join(dataset_dir, name) for name in name_list]
    train_tfrecord_file = tfrecord_file[:270]
    valid_tfrecord_file = tfrecord_file[270:]
    train(train_tfrecord_file, valid_tfrecord_file, batch_size=3,num_epochs=10)





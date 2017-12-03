import numpy as np
import tensorflow as tf
from jd_competition.array_tfrecord import prepare_batch_data

def run_eval(sess, eval_ops, num_batches_per_epoch):
   eval_metrics_tmp = []
   eval_metrics = {}
   count = 0
   while True:
       try:
           # print('eval_ops', eval_ops)
           _metrics = sess.run(eval_ops)
           # print('metrics', _metrics)
           eval_metrics_tmp.append(_metrics)
           count += 1
           if count == num_batches_per_epoch:
               break
       except tf.errors.OutOfRangeError:
           break
   # print eval_metrics_tmp
   keys = list(eval_metrics_tmp[0].keys())
   try:
       keys.remove('train_op')
   except ValueError:
       pass

   for key in keys:
       eval_metrics[key] = np.mean([x[key] for x in eval_metrics_tmp])

   # print('num_batches_eval: {0}'.format(count))
   return eval_metrics
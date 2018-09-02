import numpy as np
import tensorflow as tf
import time

v_k,v_n = 50,250000000

a = tf.placeholder(tf.float32)
size = tf.placeholder(tf.int32)
replace = tf.placeholder(tf.bool)
p = tf.placeholder(tf.float32)

y = tf.py_func(np.random.choice, [a, size, replace], tf.float32)

start = time.time()
with tf.Session() as sess:
    print(sess.run(y, {a: range(v_n), size: v_k, replace:False}))
end = time.time()
print('Spend:',str(end-start)+'s')

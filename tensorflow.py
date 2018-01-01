import tensorflow as tf


#tensors
hello = tf.constant('Hello')
type(hello) #tensorflow.python.framework.ops.Tensor

world = tf.constant('World')
result = hello + world

result #<tf.Tensor 'add:0' shape=() dtype=string>

with tf.Session() as sess:
    result = sess.run(hello+world)

result #b'HelloWorld'


#Operations

const = tf.constant(10)
fill_mat = tf.fill((4,4),10)
myzeros = tf.zeros((4,4))
myones = tf.ones((4,4))
myrandn = tf.random_normal((4,4),mean=0,stddev=0.5)
myrandu = tf.random_uniform((4,4),minval=0,maxval=1)
my_ops = [const,fill_mat,myzeros,myones,myrandn,myrandu]

"""
Interactive Session
Useful for Notebook Sessions Only
"""

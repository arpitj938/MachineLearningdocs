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
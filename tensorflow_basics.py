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

################################################################################################
#Operations


const = tf.constant(10)
fill_mat = tf.fill((4,4),10)
myzeros = tf.zeros((4,4))
myones = tf.ones((4,4))
myrandn = tf.random_normal((4,4),mean=0,stddev=0.5)
myrandu = tf.random_uniform((4,4),minval=0,maxval=1)
my_ops = [const,fill_mat,myzeros,myones,myrandn,myrandu]

"""
Tip: Interactive Session is Useful for Notebook Sessions Only
"""
################################################################################################
"""
Simple Neural Network 

"""

n_features = 10
n_dense_neurons = 3


# Placeholder for x
x = tf.placeholder(tf.float32,(None,n_features))

# Variables for w and b
b = tf.Variable(tf.zeros([n_dense_neurons]))
W = tf.Variable(tf.random_normal([n_features,n_dense_neurons]))

xW = tf.matmul(x,W)
z = tf.add(xW,b)

# tf.nn.relu() or tf.tanh()
a = tf.sigmoid(z)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    
    layer_out = sess.run(a,feed_dict={x : np.random.random([1,n_features])})

################################################################################################
""" 
Regression Example 

"""
#Artifical Data
x_data = np.linspace(0,10,10) + np.random.uniform(-1.5,1.5,10)
y_label = np.linspace(0,10,10) + np.random.uniform(-1.5,1.5,10)

plt.plot(x_data,y_label,'*')

m = tf.Variable(0.39)
b = tf.Variable(0.2)

error = 0

for x,y in zip(x_data,y_label):    
    y_hat = m*x + b  #Our predicted value
    error += (y-y_hat)**2 # The cost we want to minimize (we'll need to use an optimization function for the minimization!)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(error)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    epochs = 100
    for i in range(epochs):
        sess.run(train)
    # Fetch Back Results
    final_slope , final_intercept = sess.run([m,b])


#Evaluation 
x_test = np.linspace(-1,11,10)
y_pred_plot = final_slope*x_test + final_intercept

plt.plot(x_test,y_pred_plot,'r')
plt.plot(x_data,y_label,'*')    

################################################################################################

""" Loading and Saving"""

#saving a model
with tf.Session() as sess:
    sess.run(init)
    epochs = 100
    for i in range(epochs):
        sess.run(train)
    # Fetch Back Results
    final_slope , final_intercept = sess.run([m,b])
    # ONCE YOU ARE DONE
    # GO AHEAD AND SAVE IT!
    # Make sure to provide a directory for it to make or go to. May get errors otherwise
    #saver.save(sess,'models/my_first_model.ckpt')
    saver.save(sess,'new_models/my_second_model.ckpt')


#Loading a Model
with tf.Session() as sess:
    # Restore the model
    saver.restore(sess,'new_models/my_second_model.ckpt')
    # Fetch Back Results
    restored_slope , restored_intercept = sess.run([m,b])
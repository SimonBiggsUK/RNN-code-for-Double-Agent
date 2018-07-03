from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
import os
import tensorflow as tf

# to make output stable across runs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

def next_dance_batch(batch_size):

    n_steps = samples.shape[1];
    
    r = np.random.rand(batch_size, 1)
    t0 = r * (samples.shape[0]-1)
    batch = t0.astype(int)

    ys = np.random.rand(batch.shape[0],samples.shape[1],samples.shape[2])
    for i in range(batch.shape[0]):
        index = batch[i,:]
        ys[i] = samples[index,:,:];

    return ys[:, :-1,:], ys[:, 1:,:]

samples = np.load('dance_data_norm_60.npy')
n_steps = samples.shape[1]-1
n_inputs = n_outputs = dim = samples.shape[2]
n_neurons = 100
n_layers = 3
n_iterations = 20000
batch_size = 100
learning_rate = 0.001

reset_graph()

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_steps, n_outputs])

##cell = tf.contrib.rnn.OutputProjectionWrapper(
##    tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons, activation=tf.nn.relu),
##    output_size=n_outputs)

layers = [tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons,activation=tf.nn.relu)
          for layer in range(n_layers)]
multi_layer_cell = tf.contrib.rnn.MultiRNNCell(layers)

##cell = tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons, activation=tf.nn.relu);

rnn_outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)

stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, n_neurons])
stacked_outputs = tf.layers.dense(stacked_rnn_outputs, n_outputs)
outputs = tf.reshape(stacked_outputs, [-1, n_steps, n_outputs]) 

loss = tf.reduce_mean(tf.square(outputs - y)) # MSE
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:
    init.run()
    for iteration in range(n_iterations):
        X_batch, y_batch = next_dance_batch(batch_size)
        sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        if iteration % 100 == 0:
            mse = loss.eval(feed_dict={X: X_batch, y: y_batch})
            saver.save(sess, "./models/my_time_series_model_"+str(iteration)+"_"+str(mse))
            #print(iteration, mse)

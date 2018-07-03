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

##samples = np.load('dance_data_norm_100.npy')
##n_steps = samples.shape[1]-1
##n_inputs = n_outputs = dim = samples.shape[2]
##n_neurons = 500
##n_iterations = 20000
##batch_size = 50
##learning_rate = 0.01
next_steps = 20
##
##reset_graph()
##
##X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
##y = tf.placeholder(tf.float32, [None, n_steps, n_outputs])
##
##cell = tf.contrib.rnn.OutputProjectionWrapper(
##    tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu),
##    output_size=n_outputs)
##
##outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

samples = np.load('dance_data_norm_60.npy')
n_steps = samples.shape[1]-1
n_inputs = n_outputs = dim = samples.shape[2]
n_neurons = 100
n_layers = 3
n_iterations = 20000
batch_size = 10000
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
    
r = np.random.rand(batch_size, 1)
t0 = r * (samples.shape[0]-1)
batch = t0.astype(int)

generated=np.zeros((0,samples.shape[2]),dtype=float);
with tf.Session() as sess:                      
    saver.restore(sess, "./models/my_time_series_model_19500_0.00027318456")
    sequence = samples[0,:,:];
    for i in range(batch.shape[0]):
        index = batch[i]
        sequence = samples[int(index),:,:];
        
        for iteration in range(next_steps):
            X_batch = np.array([sequence[-1*(n_steps):,:]])
            y_pred = sess.run(outputs, feed_dict={X: X_batch})
            row = y_pred[0,-1:,:]
            sequence=np.append(sequence,row,axis=0)
        generated=np.append(generated,sequence,axis=0)
np.save('normalised_sequence', generated)

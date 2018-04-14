import tensorflow as tf
from tensorflow.python.framework import ops
import pickle
import numpy as np
ops.reset_default_graph()

import MySQLdb
import pandas as pd
db=MySQLdb.connect(host="35.226.98.33",user="ansh",passwd="",db="stations_data")
cursor = db.cursor()
query = "SELECT * FROM agg"
cursor.execute(query)
res = cursor.fetchall()
db.close()
train_cols = ["month","date","hour","minute","id","capacity"]
predict_cols = ["bikes"]
df = pd.DataFrame([*(res)],columns=["month","date","hour","minute","id","capacity","docks","bikes"])

X = df[train_cols]
Y = df[predict_cols]

del(df)

training_epochs = 1000
batch_size = 256
n_input = 6
n_classes = 1

n_hidden_1 = 6
n_hidden_2 = 2


weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1]),name="h1"),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]),name="h2"),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]),name="hout")
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1]),name="b1"),
    'b2': tf.Variable(tf.random_normal([n_hidden_2]),name="b2"),
    'out': tf.Variable(tf.random_normal([n_classes]),name="bout")
}

#keep_prob = tf.placeholder("float")
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
layer_1 = tf.nn.sigmoid(layer_1)

layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
layer_2 = tf.nn.sigmoid(layer_2)

out_layer = tf.matmul(layer_2, weights['out']) + biases['out']

cost = tf.reduce_mean(tf.squared_difference(y,out_layer))
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(training_epochs):
        avg_cost = 0.0
        total_batch = int(len(X) / batch_size)
        x_batches = np.array_split(X, total_batch)
        y_batches = np.array_split(Y, total_batch)
        for i in range(total_batch):
            batch_x, batch_y = x_batches[i], y_batches[i]
            sess.run([optimizer], 
                            feed_dict={
                                x: batch_x, 
                                y: batch_y
                            })
    pickle.dump(sess.run(weights["h1"]),open("/home/anshgandhi16/data/weights/h1.p", "wb"))
    pickle.dump(sess.run(weights["h2"]),open("/home/anshgandhi16/data/weights/h2.p", "wb"))
    pickle.dump(sess.run(weights["out"]),open("/home/anshgandhi16/data/weights/hout.p", "wb"))
    pickle.dump(sess.run(biases["b1"]),open("/home/anshgandhi16/data/weights/b1.p", "wb"))
    pickle.dump(sess.run(biases["b2"]),open("/home/anshgandhi16/data/weights/b2.p", "wb"))
    pickle.dump(sess.run(biases["out"]),open("/home/anshgandhi16/data/weights/bout.p", "wb"))

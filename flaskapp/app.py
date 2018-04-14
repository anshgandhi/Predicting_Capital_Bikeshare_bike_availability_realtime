from flask import Flask, flash, redirect, render_template, request, session, abort

app = Flask(__name__)
 
@app.route("/")
def hello():
    stat = []
    with open('./stations.csv', 'r') as f:
        temp = f.readlines()
    for i in temp:
        stat.append(i.strip().split(","))
    return render_template('results.html',hours=list(range(24)),minutes=["{:02d}".format(i) for i in list(range(0,59,5))],stations=stat)
 

@app.route("/send",methods=['POST'])
def send():
    form_vals = ["hour","minute","station"]
    _,month,date = request.form["date_time"].split("-")
    month=int(month)
    date = int(date)
    hour = int(request.form["hour"])
    minute = int(request.form["minute"])
    stat_id = int(request.form["station"])
    
    stat = []
    with open('./stations.csv', 'r') as f:
        temp = f.readlines()
    for i in temp:
        stat.append(i.strip().split(","))

    for i,j in enumerate(stat):
        if(int(j[1])==stat_id):
            cap = stat[i][2]

    import tensorflow as tf
    from tensorflow.python.framework import ops
    import pickle
    ops.reset_default_graph()

    training_epochs = 1000
    batch_size = 64
    n_input = 6
    n_classes = 1

    n_hidden_1 = 6
    n_hidden_2 = 2


    weights = {
        'h1': tf.Variable(pickle.load(open("./../weights/h1.p", "rb"))),
        'h2': tf.Variable(pickle.load(open("./../weights/h2.p", "rb"))),
        'out': tf.Variable(pickle.load(open("./../weights/hout.p", "rb")))
    }

    biases = {
        'b1': tf.Variable(pickle.load(open("./../weights/b1.p", "rb"))),
        'b2': tf.Variable(pickle.load(open("./../weights/b2.p", "rb"))),
        'out': tf.Variable(pickle.load(open("./../weights/bout.p", "rb")))
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
    
    import numpy as np
    
    X = np.array([month,date,hour,minute,stat_id,cap]).reshape(-1,6)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        pred = sess.run([out_layer], feed_dict={x: X})
    pred_1 = pred[0][0][0]
    
    
    from sklearn.ensemble import RandomForestRegressor
    rf = pickle.load(open("./../weights/rf.p", "rb"))
    pred_2 = rf.predict(X)[0]

    pred = (pred_1+pred_2)/2
    
    return "<br>".join(["Predicted Number of Bikes are: "+"{:.0f}".format(np.round((pred))),"Capacity at Station: "+str(cap)])
    
    
    station
    
if __name__ == "__main__":
    app.run(host= '0.0.0.0')
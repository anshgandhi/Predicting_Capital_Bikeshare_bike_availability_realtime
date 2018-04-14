from flask import Flask, flash, redirect, render_template, request, session, abort

app = Flask(__name__)
 
@app.route("/")
def hello():
    import pandas as pd
    stat = []
    with open('./stations.csv', 'r') as f:
        temp = f.readlines()
    for i in temp:
        stat.append(i.strip().split(","))
    
    import xarray as xr
    import numpy as np
    import pandas as pd
    import holoviews as hv
    import geoviews as gv
    import geoviews.feature as gf
    import cartopy
    from cartopy import crs as ccrs
    from bokeh.models import WMTSTileSource
    from bokeh.models import HoverTool
    from bokeh.plotting import figure, show, output_notebook
    from bokeh.embed import components
    from bokeh.resources import INLINE
    
    # grab the static resources
    js_resources = INLINE.render_js()
    css_resources = INLINE.render_css()
    
    stat = []
    with open('./stations.csv', 'r') as f:
        temp = f.readlines()
    for i in temp:
        stat.append(i.strip().split(","))
    import pandas as pd
    df = pd.DataFrame(stat,columns=["StationName","Id","Capacity","Lat","Long"])
    hover = HoverTool(tooltips=[
        ("Station Name", "@StationName")
    ])
    plot_width = 1200
    plot_height = 400

    # IP Address - Instance 1 (t2.medium)
    hv.renderer('bokeh')

    # World Map tile option 1
    CARTODBPOSITRON_RETINA = WMTSTileSource(
        url='http://tiles.basemaps.cartocdn.com/light_all/{z}/{x}/{y}@2x.png',
        attribution=(
            '&copy; <a href="http://www.openstreetmap.org/copyright">OpenStreetMap</a>'
            ' contributors,'
            '&copy; <a href="https://cartodb.com/attributions">CartoDB</a>'
        )
    )
    plot_df = pd.DataFrame()
    plot_df["Long"] = df["Long"].values.astype("float")
    plot_df["Lat"] = df["Lat"].values.astype("float")
    plot_df["StationName"] = df["StationName"]
    population = gv.Dataset(plot_df, kdims=["Long", "Lat"],vdims=["StationName"])

    layout = (gv.WMTS(CARTODBPOSITRON_RETINA, extents=(0,0, 0, 0),crs=ccrs.PlateCarree())*\
              population.to(gv.Points,kdims=["Long", "Lat"],vdims=["StationName"],crs=ccrs.PlateCarree()))\
    (style={"Points":dict(size=1,fill_alpha=0.4,line_alpha=0)})\
    (plot={'Overlay':dict(width=plot_width,height=plot_height,show_legend=False),'Points':dict(tools=[hover],xaxis=None,yaxis=None,colorbar=False)})
    renderer = hv.renderer('bokeh')
    doc = renderer.server_doc(layout)
    script, div  = components(doc)
    return render_template('results.html',js_resources=js_resources,css_resources=css_resources,the_div=div,the_script=script,hours=list(range(24)),minutes=["{:02d}".format(i) for i in list(range(0,59,5))],stations=df[["StationName","Id"]].values.tolist())
 

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
    pred = np.round((pred[0][0][0]))
    
    return "<br>".join(["Predicted Number of Bikes are: "+"{:.0f}".format(np.round((pred))),"Capacity at Station: "+str(cap)])
    
    
    station
    
if __name__ == "__main__":
    app.run(host= '0.0.0.0')
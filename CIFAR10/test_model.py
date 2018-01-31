import tensorflow as tf
import numpy as np
from IPython.display import clear_output, Image, display, HTML
import pdb
from data import get_data_set
from tensorflow.python.tools import inspect_checkpoint as chkp

_BATCH_SIZE = 50

def strip_consts(graph_def, max_const_size=32):
    """Strip large constant values from graph_def."""
    strip_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = strip_def.node.add() 
        n.MergeFrom(n0)
        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            size = len(tensor.tensor_content)
            if size > max_const_size:
                tensor.tensor_content = "<stripped %d bytes>"%size
    return strip_def

def show_graph(graph_def, max_const_size=32):
    """Visualize TensorFlow graph."""
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()
    strip_def = strip_consts(graph_def, max_const_size=max_const_size)
    code = """
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:600px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(str(strip_def)), id='graph'+str(np.random.rand()))

    iframe = """
        <iframe seamless style="width:1200px;height:620px;border:0" srcdoc="{}"></iframe>
    """.format(code.replace('"', '&quot;'))
    display(HTML(iframe))

x = tf.placeholder(tf.float32, shape=[None, 3072])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
keep_prob2 = tf.placeholder(tf.float32)
keep_prob3 = tf.placeholder(tf.float32)
keep_prob4 = tf.placeholder(tf.float32)
keep_prob5 = tf.placeholder(tf.float32)
keep_prob6 = tf.placeholder(tf.float32)
correct_prediction = tf.placeholder(tf.bool)

sess=tf.Session()    
#First let's load meta graph and restore weights
saver = tf.train.import_meta_graph('./models/-0.meta')
saver.restore(sess, "./models/-0")

test_x, test_y, test_l = get_data_set(name="test", cifar=10, aug=False)
test_correct_pred = np.zeros(shape=len(test_x), dtype=np.int)

for i in range(len(test_x)/_BATCH_SIZE):
    test_batch_x = test_x[i*_BATCH_SIZE:(i+1)*_BATCH_SIZE-1, :]
    test_batch_y = test_y[i*_BATCH_SIZE:(i+1)*_BATCH_SIZE-1, :]
    test_correct_pred[i*_BATCH_SIZE:(i+1)*_BATCH_SIZE-1] = correct_prediction.eval(session=sess,
                                                                                 feed_dict={x: test_batch_x, y_: test_batch_y, keep_prob2: 1.0, keep_prob3: 1.0,
                                                                                            keep_prob4: 1.0, keep_prob5: 1.0, keep_prob6: 1.0})

# Test accuracy computation
test_accuracy = test_correct_pred.mean() * 100
correct_num = test_correct_pred.sum()
print("Accuracy on Test Set: {0:.2f}% ({1} / {2})".format(test_accuracy, correct_num, len(test_x)))

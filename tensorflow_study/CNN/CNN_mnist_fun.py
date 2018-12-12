
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import tensorboard as tbd
mnist=input_data.read_data_sets('MNIST_DATA_BAK',one_hot=True)
input_shape=np.array(mnist.train.images).reshape(-1,28,28,1)
out_shape=np.array(mnist.train.labels).shape[1]

with tf.name_scope('input'):
    X=tf.placeholder(shape=(None,28,28,1),dtype=tf.float32)
    y=tf.placeholder(shape=(None,out_shape),dtype=tf.float32)

def weight(shape):
    return tf.Variable(tf.truncated_normal(shape=shape,stddev=0.1))
def bias(shape):
    b_=tf.ones(shape=shape)
    return tf.Variable(b_)
with tf.name_scope('cnn1'):
    w_cnn1=weight(shape=(3,3,1,24))
    b_cnn1=bias(shape=(24))
    cnn1=tf.nn.conv2d(input=X,filter=w_cnn1,strides=[1,1,1,1],padding='SAME')+b_cnn1
    relu1=tf.nn.relu(cnn1)
    pool1=tf.nn.max_pool(relu1,ksize=(1,2,2,1),strides=(1,2,2,1),padding='SAME')
with tf.name_scope('cnn2'):
    w_cnn2=weight(shape=(3,3,24,48))
    b_cnn2=bias(shape=(48))
    cnn2=tf.nn.conv2d(input=pool1,filter=w_cnn2,strides=[1,1,1,1],padding='SAME')+b_cnn2
    relu2=tf.nn.relu(cnn2)
    pool2=tf.nn.max_pool(relu2,ksize=(1,2,2,1),strides=(1,2,2,1),padding='SAME')

pool2_reshape=tf.reshape(pool2,shape=(-1,7*7*48))
with tf.name_scope('fc1'):
    w_fc1=weight(shape=(7*7*48,100))
    b_fc1=bias(shape=(100))
    fc1=tf.matmul(pool2_reshape,w_fc1)+b_fc1
    relu_fc1=tf.nn.relu(fc1)
with tf.name_scope('softmax'):
    w_softmax=weight(shape=(100,10))
    b_sotfmax=bias(shape=(10))
    #注意，这一步不能少，即不能缺失tf.nn.softmax()这个函数，
    softmax=tf.nn.softmax(tf.matmul(relu_fc1,w_softmax)+b_sotfmax)
with tf.name_scope('loss'):
    loss=tf.reduce_mean(tf.reduce_sum(-y*tf.log(softmax),reduction_indices=[1]))

with tf.name_scope('train'):
    # optimizer=tf.train.GradientDescentOptimizer()

    optimizer=tf.train.AdamOptimizer()
    train=optimizer.minimize(loss=loss)

with tf.name_scope('acc'):
    correct=tf.equal(tf.arg_max(softmax,1),tf.arg_max(y,1))
    correct=tf.cast(correct,dtype=tf.float32)
    acc=tf.reduce_mean(correct)

with tf.Session() as sess:
    writer=tf.summary.FileWriter('log/',sess.graph)
    init=tf.global_variables_initializer()
    init.run()
    for i in range(1000):
        X_batch,y_batch=mnist.train.next_batch(50)
        X_batch=np.array(X_batch).reshape(-1,28,28,1)
        train.run(feed_dict={X:X_batch,y:y_batch})
        if i%100==0 and i<1000:
            print('train acc:',i,acc.eval(feed_dict={X:X_batch,y:y_batch}))
    X_batch= mnist.test.images
    X_batch = np.array(X_batch).reshape(-1, 28, 28, 1)
    y_batch=mnist.test.labels
    print('test acc:',acc.eval(feed_dict={X:X_batch,y:y_batch}))
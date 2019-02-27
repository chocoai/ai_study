import tensorflow as tf
from tensorflow.contrib import rnn
lstm_num_units_encoder=10
lstm_num_units_decoder=10
#分类的种类数量
n_classes=10
#一个音频文件的大小
diminput=100
#一个音频文件被切分成多少份
nsteps=10

learning_rate=0.001

x=tf.placeholder(shape=(None,diminput))
y=tf.placeholder(shape=(None,n_classes))

W={'w_decoder':tf.Variable(tf.random_normal(shape=(lstm_num_units_encoder,lstm_num_units_decoder))),
   'w_sotfmax':tf.Variable(tf.random_normal(shape=(lstm_num_units_decoder,n_classes)))}
B={'b_decoder':tf.Variable(tf.zeros(shape=(lstm_num_units_decoder))),
   'b_sotfmax':tf.Variable(tf.zeros(shape=(n_classes)))}
def RNN(x,nsteps):
    x=tf.split(x,nsteps)
    lstm_cell=tf.nn.rnn_cell.BasicLSTMCell(num_units=lstm_num_units_encoder)
    LSTM_O, LSTM_S = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    return {'LSTM_O':LSTM_O,'LSTM_S':LSTM_S}

myrnn=RNN(x,nsteps=nsteps)

#decoder  目前只有一层，所以直接用全连接+softmax，后续可以如法套用lstm
decoder_out=tf.matmul(myrnn['LSTM_O'][-1],W['w_decoder'])+B['b_decoder']

sotfmax_out=tf.matmul(decoder_out,W['w_sotfmax'])+B['b_sotfmax']

loss=tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=sotfmax_out)
optm=tf.train.AdamOptimizer(learning_rate=learning_rate)
opt=optm.minimize(loss=loss)

ini=tf.global_variables_initializer()

with tf.Session() as sess:
    fw=tf.summary.FileWriter(logdir='logs/',graph=sess.graph)
    ini.run()
    #todo
    opt.run(feed_dict={})
'''注意，这里先做一些简化处理，如：
1.encoder过程中的输出只有一个，后续可以将输入作为一个向量
2.decoder过程，暂只做一层，并在这一层做softmax分类预测

问题:
没太搞明白这个损失函数及对变量的求偏导，后面可以接入tensorflow的方法，构建lstm并用optimizer计算、求解模型

'''
from tensorflow.contrib import rnn
import tensorflow as tf
import numpy as np

m=10#输入的数据个数,这里是指进行训练的语音个数
m_x=10#一个语音可以截取成多少个20ms的输入单元
n=10#20ms长度的数据向量维度数量
n_rnn_encoder_layer=10#rnn encoder层的个数
n_rnn_decoder_layer=1#rnn decoder层的个数
n_out_rnn=1#rnn的局部输出元素个数
n_out_rnn_decoder=1

rnn_bias_size=1
#这里的一个x表示，20ms长度的数据？这里需要对一个语音进行处理，以便构建神经网络
x=tf.placeholder(dtype=tf.float32,shape=(m,n))
# one hot encoder for y
y=tf.placeholder(dtype=tf.float32,shape=(m,10))

theta_rnn_encoder=tf.Variable(tf.ones(shape=(n+1,1)))
theta_rnn_decoder=tf.Variable(tf.ones(shape=(1,n_out_rnn_decoder)))
theta_softmax_decoder=tf.Variable(tf.ones(shape=(n+1,1)))
b_rnn_encoder=tf.Variable(tf.constant(0))
b_rnn_decoder=tf.Variable(tf.constant(0))
b_softmax_decoder=tf.Variable(tf.constant(0))

out_rnn_encoder=np.mat(np.ones((n_out_rnn,1)))

#need to replace this x_batch 当然前提是将数据进行清洗，玩转
def gen_rnn_encoder_layer(x_batch, theta_rnn_encoder, b_rnn_encoder, out_rnn_encoder):
    x_=np.vstack((out_rnn_encoder,x))
    y_=tf.matmul(x_,theta_rnn_encoder)+b_rnn_encoder
    y_hat=tf.nn.sigmoid(y_)
    out_rnn_encoder=y_hat
    return out_rnn_encoder

for i in range(n_rnn_encoder_layer):
    out_rnn_encoder=gen_rnn_encoder_layer(x, theta_rnn_encoder, b_rnn_encoder, out_rnn_encoder)

#decoder
out_rnn_decoder=out_rnn_encoder
def gen_gen_rnn_decoder_layer(out_rnn_decoder,theta_rnn_decoder,b_rnn_decoder):
    y_=tf.matmul(out_rnn_decoder,theta_rnn_decoder)+b_rnn_decoder
    y_hat=tf.nn.sigmoid(y_)
    out_rnn_decoder=y_hat
    return out_rnn_decoder

y_=[]
for i in range(n_rnn_decoder_layer):
    out_rnn_decoder=gen_gen_rnn_decoder_layer(out_rnn_decoder,theta_rnn_decoder,b_rnn_decoder)
    y_.append(tf.nn.softmax(out_rnn_decoder))

#暂只考虑一层decoder的情况，loss不用连乘
loss=tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=y_[0])

optm=tf.train.AdamOptimizer(learning_rate=0.0001)
opt=optm.minimize(loss)

ini=tf.global_variables_initializer()
with tf.Session() as sess:
   ini.run()


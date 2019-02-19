'''注意，这里先做一些简化处理，如：
1.encoder过程中的输出只有一个，后续可以将输入作为一个向量
2.decoder过程，暂只做一层，并在这一层做softmax分类预测

问题:
没太搞明白这个损失函数及对变量的求偏导，后面可以接入tensorflow的方法，构建lstm并用optimizer计算、求解模型

'''
from tensorflow.contrib import rnn
import tensorflow as tf
import numpy as np
m=10
n=10

rnn_bias_size=1

x=tf.placeholder(dtype=tf.float32,shape=(m,n))
# one hot encoder for y
y=tf.placeholder(dtype=tf.float32,shape=(m,10))

theta_rnn_encoder=tf.Variable(tf.ones(shape=(n+1,1)))
b_rnn_encoder=tf.Variable(tf.constant(0))

out_rnn_encoder=np.mat(np.ones((m,1)))



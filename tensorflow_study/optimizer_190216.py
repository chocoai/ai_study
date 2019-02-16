import tensorflow as tf
from sklearn.datasets import fetch_california_housing
import numpy as np

n_epochs=50000
learning_rate=0.01
housing_data = fetch_california_housing(data_home='D:\\学习笔记\\ai\\dataSets', download_if_missing=True)

#data  target feature_names
data=housing_data['data']
bia=np.float32((np.ones(shape=(data.shape[0],1))))
data_bia=np.float32(np.hstack((bia,data)))

target=housing_data['target']
target=np.float32(target.reshape(-1,1))

#tensoflow
x=tf.constant(data_bia,dtype=tf.float32)
y=tf.constant(target,dtype=tf.float32)

theta=tf.Variable(tf.ones(shape=(data_bia.shape[1],1)),dtype=tf.float32)
# theta = tf.Variable(tf.random_uniform([data_bia.shape[1], 1], -1.0, 1.0), name='theta')

y_hat=tf.matmul(x,theta)
erro=y-y_hat
loss=tf.reduce_mean(tf.square(erro))

'''
用梯度下降会导致梯度爆炸？从而引起损失不减小反而增加？有可能是因为记录数量太大了，导致在计算的过程中数据溢出
用SGD或mini batch GD可能会正常
'''
# optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01,name='GradientDescentOptimizer')
optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate,name='GradientDescentOptimizer')
optimizer_opt=optimizer.minimize(loss)

init_g = tf.global_variables_initializer()

with tf.Session() as sess:
    init_g.run()
    for n_epoch in range(n_epochs):
        optimizer_opt.run()
        learning_rate=learning_rate*learning_rate/(learning_rate+n_epoch)
        if(n_epoch%100==0):
            # print(erro)
            print(loss.eval())

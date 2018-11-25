from sklearn.datasets import fetch_california_housing
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

n_epochs = 100000
learning_rate = 0.001
housing_data = fetch_california_housing(data_home='D:\\学习笔记\\ai\\dataSets', download_if_missing=True)
data = housing_data.data
print(housing_data.feature_names)
m, n = data.shape
# 这一步numpy是直接执行的，不能放到tf里面去执行吗？
housing_data_puls_bias = np.c_[np.ones((m, 1)), data]

X = tf.constant(housing_data_puls_bias, name='X', dtype=tf.float32)
y = tf.constant(housing_data.target.reshape(-1, 1), name='y', dtype=tf.float32)
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name='theta')
print('theta:', theta.shape)
y_hat = tf.matmul(X, theta, name='y_hat')
err = y_hat - y
print(y_hat.shape, y.shape)
print('err.shape:', err.shape)

mse = tf.reduce_mean(tf.square(err), name='mse')
gradient = 2 / m * tf.matmul(tf.transpose(X), err)
theta_1=theta - tf.Variable(learning_rate * gradient)
train_op = tf.assign(theta, theta_1)

init = tf.global_variables_initializer()
print(X.shape)

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        if epoch%100==0:
            # print('theta', theta.eval())
            print('mse.eval():',mse.eval())
            # print(gradient.eval())
            train_op.eval()
    print('theta:', theta.eval())

'''
在图形识别时，将图片某一小块作为一个部分，来提取特征，要比直接转换位一维数组要好
自然而然的，就可以想到这样来做（仍然离不开算法来源于思想）
卷积核是怎么想出来的？卷积核是对图片不同角度的审视

'''
from sklearn.datasets import load_sample_images
import numpy as np
import PIL
import tensorflow as tf
import matplotlib.pyplot as plt

dataset=np.array(load_sample_images()['images'])
print(dataset.shape)
batch_size,h,w,channels=dataset.shape
filters=np.zeros(shape=(7,7,channels,2),dtype=np.float32)
filters[3,:,:,0]=1
filters[:,3,:,1]=1

X=tf.placeholder(shape=dataset.shape,name='X',dtype=np.float32)
convolution=tf.nn.conv2d(input=X,filter=filters,strides=[1,2,2,1],padding='SAME')

with tf.Session() as sess:
    output=sess.run(convolution,feed_dict={X:dataset})

n=0
plt.imshow(dataset[n])
plt.show()
plt.imshow(output[n,:,:,1])
plt.show()
plt.imshow(output[n,:,:,0])
plt.show()

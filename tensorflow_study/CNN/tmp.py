
import tensorflow as tf
from sklearn.datasets import load_sample_images
import numpy as np
import matplotlib.pyplot as plt
data=np.array(load_sample_images()['images'])
batch_size,height,width,channels=data.shape
X=tf.placeholder(shape=data.shape,name='X',dtype=np.float32)
filter=np.zeros(shape=(7,7,channels,1))
filter[2,:,:,:]=1
convolution=tf.nn.conv2d(input=X,filter=filter,strides=[1,2,2,1],padding='SAME')

with tf.Session() as sess:
    output=sess.run(convolution,feed_dict={X:data})

plt.imshow(data[0])
plt.show()
# print(output.shape)
plt.imshow(output[0,:,:,0])
plt.show()
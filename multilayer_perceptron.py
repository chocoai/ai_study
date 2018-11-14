import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
# //作业是MLPRegressor预测水泥强度
#ps 画出神经网络拓扑图
from sklearn.neural_network import MLPRegressor
'''
X=[[1,1],[0,0],[3,4],[5,0]]
Y=[1,0,0,1]
clf=MLPClassifier(hidden_layer_sizes=(2,2),activation='logistic',
                  alpha=0.0001,max_iter=10000,solver='sgd')
clf.fit(X,Y)

X_test=[[1,0],[5,1]]
Y_predict=clf.predict(X_test)
Y_proba=clf.predict_proba(X_test)
print(Y_predict)
print(Y_proba)
print([coef.shape for coef in clf.coefs_])
print([coef for coef in clf.coefs_])
'''

# 开始预测波士顿房价
print('HousePrice')
data=pd.read_csv('HousePrice.csv')

data=np.array(data)
data_test=data[271:,:]
data=data[:270,:]#270 for train
X=data[:,:4]
Y=data[:,13]#注意，这里的13后面一定不能加冒号！！！否则Y会变成2D矩阵，交给模型算的时候就完全不一样了！！！
model=MLPRegressor(hidden_layer_sizes=(10,10,10),activation='relu',solver='sgd',max_iter=1000)
model.fit(X,Y)
# print('coef.shape',[coef.shape for coef in model.coefs_])
# print([coef for coef in model.coefs_])

X_test=data_test[:,:4]
Y_test=data_test[:,13]

Y_predict=model.predict(X_test)

print(Y_test)
print('Y_predict',Y_predict)
# print(np.corrcoef(Y_test,Y_predict))
# print(Y.shape)
# 问题大大的啊！！！！！
# 为什么计算出来差距这么大呢？！



'''
# 由于出现了预测结果完全一样的异常，所以先用简单的数据进行验证
print('MLPRegressor for easier')
X=[[1,1],[0,0],[3,4],[5,0]]
Y=[1,3,0,1]
clf=MLPRegressor(hidden_layer_sizes=(2,2),activation='logistic',
                  alpha=0.0001,max_iter=10000,solver='sgd')
clf.fit(X,Y)

X_test=[[1,0],[5,1]]
Y_predict=clf.predict(X_test)
print(Y_predict)
print([coef.shape for coef in clf.coefs_])
print([coef for coef in clf.coefs_])
'''
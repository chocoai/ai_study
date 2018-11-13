
from sklearn.neural_network import MLPClassifier
# //作业是MLPRegressor预测水泥强度
#ps 画出神经网络拓扑图
from sklearn.neural_network import MLPRegressor
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
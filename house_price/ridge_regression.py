from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures,StandardScaler,OrdinalEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from numpy import  *

scaler = StandardScaler()
data=pd.read_csv('house_price.csv',encoding = "utf-8")
data=np.array(data)
y=data[:,3:4]

enc = OrdinalEncoder()
enc.fit(data[:,2:3])
data_enc=enc.transform(data[:,2:3])
# print('data_enc:',data_enc)
data=hstack((data[:,:2],data_enc))

X=data[:,:3]
# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=23)
X_train,X_test,y_train,y_test=X[:300,:],X[300:,:],y[:300,:],y[300:,:]
print('X_train:',hstack((X_train,y_train)))
# print('y_train:',y_train)

# scaler.fit(X)
# X = scaler.transform(X)
# model=Ridge(solver='sag',random_state=23)
# model.fit(X,y)
# y_predict=model.predict(X_test)
# print(y_predict)

poly_features=PolynomialFeatures(degree=3,include_bias=False)
X_poly=poly_features.fit_transform(X_train)
# print(X_poly)
model2=Ridge(solver='sag',random_state=23,tol=1e-4,max_iter=10000)
model2.fit(X_poly,y_train)
X_test=poly_features.fit_transform(X_test)
y_predict=model2.predict(X_poly)

print(hstack((y_train,y_predict)))

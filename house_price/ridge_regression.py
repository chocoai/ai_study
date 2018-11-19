from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
import pandas as pd
import numpy as np

scaler = StandardScaler()
data=pd.read_csv('house_price.csv',encoding = "utf-8")
data=np.array(data)
X=data[:,:3]
y=data[:,3:4]

# scaler.fit(X)
# X = scaler.transform(X)
X_test=data[:20,:3]
y_test=y[:20,:]

# model=Ridge(solver='sag',random_state=23)
# model.fit(X,y)

# y_predict=model.predict(X_test)
# print(y_predict)

poly_features=PolynomialFeatures(degree=3,include_bias=False)
X_poly=poly_features.fit_transform(X)
# print(X_poly)
model2=Ridge(solver='sag',random_state=23,tol=1e-4)
model2.fit(X_poly,y)
X_test=poly_features.fit_transform(X_test)
y_predict=model2.predict(X_test)
print(y_predict,y_test)

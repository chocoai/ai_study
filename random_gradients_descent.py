import numpy as np

X=np.random.rand(100,1)
X_b=np.c_[np.ones((100,1)),X]
Y=5+7*X+np.random.randn(100,1)


flag=False
n_epochs=2000
m=100
learn_rate=0.1
t0,t1=10,50
def learning_schedule(t):
    return t0/(t1+t)

theta=np.random.randn(2,1)

for epoch in range(n_epochs):
    for j in range(m):
        #random line
        random_index=np.random.randint(m)
        xi=X_b[random_index:random_index+1]
        yi=Y[random_index:random_index+1]
        gradient=xi.T.dot(xi.dot(theta)-yi)

        learn_rate=j+epoch*m
        learn_rate=learning_schedule(learn_rate)
        theta=theta-gradient*learn_rate
        epsi=X_b.dot(theta)-Y
print(theta)
print(flag)
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model, metrics
from linear_regression import linear_regression_model
TEST_N = 20
x = int(input())
arr = []
for i in range(x):
		x = np.random.uniform(0,300)
		y = np.random.uniform(0,300)
		z = (x + y) / 2 + np.random.uniform(-25,25)
		arr.append([x,y,z])
arr = np.asarray(arr)
X, Y = arr[:, [0,1]], arr[:, 2]
reg = linear_model.LinearRegression()
#X = X.reshape(-1,1)
Y = Y.reshape(-1,1)
reg.fit(X, Y)
#coef = reg.coef_
#coef = coef.flatten()
print('Coefficients: \n', reg.coef_)
eval = []
for i in range(TEST_N):
		x = np.random.randint(30,270)
		y = np.random.uniform(30,270)
		z = (x + y) / 2 + np.random.uniform(-2,6)
		eval.append([x,y,z])
eval = np.asarray(eval)
XX, YY = eval[:, [0,1]], eval[:, 2] #for 3d 
#XX = XX.reshape(-1,1)
YY = YY.reshape(-1,1)

print('Variance score: {}'.format(reg.score(XX, YY))) # 1 for perfect, less is worse

res = []
res.append(reg.predict(XX))
res = np.asarray(res)
res = res.reshape(-1,1)
an_array = np.append(res, YY, axis=1)
print (an_array)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(X[:, 0], X[:, 1], Y, color='green');
ax.scatter3D(XX[:, 0], XX[:, 1], YY, color='red');
#plt.show()

print (X.shape)
print (Y.shape)
print (XX.shape)
print (YY.shape)
X = X.T
Y = Y.T
XX = XX.T
YY = YY.T

print (X.shape)
print (Y.shape)
print (XX.shape)
print (YY.shape)

#linear_regression_model(X.T, Y.T, XX.T, YY.T, 0.01, 10)

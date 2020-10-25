import numpy as np
import matplotlib.pyplot as plt

#step1
def initialize_parameters(lenw):
		w = np.random.randn(1,lenw)
		b = 0
		return w, b

#ste2
def forward_prop(X, w, b):	#w -> 1xm, X -> nxm
		z = np.dot(w,X) + b	#z -> 1xm b_vector = [b b b ...]
		return z

#step3
def cost_function(z, y):
		m = y.shape[1]
		J = (1/(2*m)) * np.sum(np.square(z-y))
		return J

#step4
def back_prop(X, y, z):
		m = y.shape[1]
		dz = (1/m) * (z-y)
		dw = np.dot(dz,X.T)
		db = np.sum(dz)
		return dw, db

#step5
def gradiant_descent_update(w, b, dw, db, learning_rate):
		w = w - learning_rate * dw
		b = b - learning_rate * db
		return w, b

#step6
def linear_regression_model(x_train, y_train, x_val, y_val, learning_rate, epochs):
		lenw = x_train.shape[0]
		w, b = initialize_parameters(lenw) #step1
		costs_train = []
		m_train = y_train.shape[1]
		m_val = y_val.shape[1]

		for i in range(1, epochs + 1):
				z_train = forward_prop(x_train, w, b) #step2
				cost_train = cost_function(z_train, y_train) #step3
				dw, db = back_prop(x_train, y_train, z_train) #step4
				w, b = gradiant_descent_update(w, b, dw, db, learning_rate) #step5
				#store for plotting purpose
				if (i % 1 == 0):
						costs_train.append(cost_train)

				#MAE train
				MAE_train = (1/m_train) * np.sum(np.abs(z_train-y_train))

				#cost_val, MAE val
				
				z_val = forward_prop(x_val, w, b)
				cost_val = cost_function(z_val, y_val)
				MAE_val = (1/m_val)*np.sum(np.abs(z_val-y_val))

				#print out cost_train, cost_val, MEE_train, MAE_val
				print ('Epochs '+str(i)+'/'+str(epochs)+': ')
				print ('Training cost '+str(cost_train)+' | '+'Validation cost '+str(cost_val))
				print ('MAE train '+str(MAE_train)+' | '+'MAE_val '+str(MAE_val))

		experimental = 0
		for i in range(len(z_val[0])):
				experimental += (y_val[0][i] - z_val[0][i])/y_val[0][i]
		np.set_printoptions(precision=1)
		print (1 - experimental/len(z_val[0]))
		print (z_val)
		
#		print (dw)
#		print (db)
		plt.plot(costs_train)
#		plt.scatter(x_train, y_train, color="red")
#		plt.scatter(x_val, y_val, color="blue")
		plt.xlabel('Iterations(per_tens)')
		plt.ylabel('Training cost')
		plt.title('Learning rate '+str(learning_rate))
#		plt.show()
'''
x_train = [[1,1], [2,2], [3,3], [4,4], [5,5], [6,6], [7,7], [8,8], [9,9], [10,10]]
y_train = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
x_val = [[5.5,5.5], [6.5,6.5], [7.5,7.5]]
y_val = [5.5, 6.5, 7.5]
x_train = np.asarray(x_train)
y_train = np.asarray([y_train])
x_val = np.asarray(x_val)
y_val = np.asarray([y_val])
x_train = x_train.T
x_val = x_val.T
print (x_train.shape)
print (y_train.shape)
print (x_val.shape)
print (y_val.shape)
print (y_val.shape[1])
print (y_val.T)
#zzz = y_val.T
#print (zzz.shape)
#exit()
'''
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

import pandas as pd

boston = load_boston()
bost = pd.DataFrame(boston['data'])
bost.columns = boston['feature_names']
X = (bost - bost.mean())/(bost.max()-bost.min())
Y = boston['target']
#print (X.shape)
x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size = 0.02, random_state = 5)
y_train = np.asarray([y_train])
y_val = np.asarray([y_val])
x_train = x_train.T
x_val = x_val.T
#print(x_train.shape)
#print(y_train.shape)
linear_regression_model(x_train, y_train, x_val, y_val, 0.1, 10000)
print (y_val)

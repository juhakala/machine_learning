import numpy as np
from logistic_model import logistic_regression
import matplotlib.pyplot as plt

SPLIT = 0.7
# generate random input data arr[0-1] and correct output data y
arr = np.random.uniform(0,1,(1,2))
y = np.array([False if (arr[0][0] + arr[0][1]) / 2 > SPLIT else True])
x = int(input())
for i in range(1,x):
		arr = np.append(arr, np.random.uniform(0,1,(1,2)), axis=0)
		y = np.append(y, [False if (arr[i][0] + arr[i][1]) / 2 > SPLIT else True])

#fetch logistic model and drive
model = logistic_regression(lr=0.01, num_iter=100000) #change learning rate and number of iterations to see what fits
model.fit(arr,y)

#plotting
plt.style.use('fivethirtyeight')
#plt.plot([0, 1], [1, 0], color='k', linestyle='-', linewidth=1)
colors = ["r" if x == True else "b" for x in y]
plt.scatter(arr[:, 0], arr[:,1], color = colors, s = 5, label = 'Train data')

# generate random test input data arr[0-1] and correct test output data y
arr = np.random.uniform(0,1,(1,2))
y = np.array([False if (arr[0][0] + arr[0][1]) / 2 > SPLIT else True])
for i in range(1,30):
		arr = np.append(arr, np.random.uniform(0,1,(1,2)), axis=0)
		y = np.append(y, [False if (arr[i][0] + arr[i][1]) / 2 > SPLIT else True])


#test how "Kyolevi" learned
acc = model.predict(arr, 0.5)
print ("predict: ", acc)
print ("actual:  ", y)
print ("accuracy: ", (acc == y).mean())
print (model.theta)

print (arr[0])
print (arr[1])

colors = ["orange" if x == True else "green" for x in acc]
plt.scatter(arr[:, 0], arr[:,1], color = colors, s = 20, label = 'test data')

plt.show()


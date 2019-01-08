import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("This script takes data.csv as default data")
 
alpha=float(input("Enter the value of alpha: "))
iterations=int(input("Enter number of iterations: "))
x_label=input("Enter x label: ")
y_label=input("Enter y label: ")

data=pd.read_csv("data.csv", header=None)

m=len(data)
x=np.hstack((data.iloc[:, 0])).reshape(m,1)
Y=np.hstack((data.iloc[:, 1])).reshape(m,1)

ones= np.ones((m,1))
X=np.hstack((ones,x))

theta= np.zeros((2,1))

def costfn(theta):
    temp=X.dot(theta)-Y
    return np.sum(np.power(temp,2))/(2*m)

def gradient(theta):
    for _ in range(iterations):
        temp=X.dot(theta)-Y
        temp2=X.transpose()
        temp=temp2.dot(temp)
        theta= theta - (alpha/m)*temp
    return theta

theta=gradient(theta)
cost_value=costfn(theta)

print("Value of theta and minimum error")
print(theta)
print(cost_value)

plt.scatter(x, Y)
plt.plot(x, X.dot(theta))
plt.xlabel(x_label)
plt.ylabel(y_label)
plt.show()

min_theta=int(np.amin(theta)-5)
max_theta=int(np.amax(theta)+5)

theta0=np.arange(min_theta,max_theta,0.1)
theta1=np.arange(min_theta,max_theta,0.1)
j_vals= np.zeros((len(theta0),len(theta1)))

t=np.ones((2,1))

for i in range(len(theta0)):
    for j in range(len(theta1)):
        t[0][0]=theta0[i]
        t[1][0]=theta1[j]
        j_vals[i][j]=costfn(t)

plt.contourf(theta0,theta1, j_vals)
plt.show()
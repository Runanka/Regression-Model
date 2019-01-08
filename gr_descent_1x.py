import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("This script takes data.csv as default data")
 
alpha=float(input("Enter the value of alpha: "))
iterations=int(input("Enter number of iterations: "))
x_label=input("Enter x label: ")
y_label=input("Enter y label: ")

data=pd.read_csv("data.csv", header=None)        

m=len(data)                                         # total data
x=np.hstack((data.iloc[:, 0])).reshape(m,1)         # creates a vector of (m,1) from 1st index
Y=np.hstack((data.iloc[:, 1])).reshape(m,1)         # creates a vector of (m,1) from 2nd index

ones= np.ones((m,1))                                # creates a vector of (m,1) of ones for x0
X=np.hstack((ones,x))                               # merge x and ones to X

theta= np.zeros((2,1))                              # theta vector of (2,1): theta0, theta1

def costfn(theta):                                  # calculates cost function
    temp=X.dot(theta)-Y
    return np.sum(np.power(temp,2))/(2*m)

def gradient_descent(theta):                                # calculates gradient descent
    for _ in range(iterations):
        temp=X.dot(theta)-Y
        temp2=X.transpose()
        temp=temp2.dot(temp)
        theta= theta - (alpha/m)*temp
    return theta

theta=gradient_descent(theta)
cost_value=costfn(theta)

print("Value of theta and minimum error")
print(theta)
print(cost_value)

plt.scatter(x, Y)                                   # scatter plot
plt.plot(x, X.dot(theta))                           # plot regression line
plt.xlabel(x_label)
plt.ylabel(y_label)
plt.show()

min_theta=int(np.amin(theta)-5)                     # creates a range for theta as input
max_theta=int(np.amax(theta)+5)

theta0=np.arange(min_theta,max_theta,0.1)           # creates theta0 from the range
theta1=np.arange(min_theta,max_theta,0.1)
j_vals= np.zeros((len(theta0),len(theta1)))         # to store the cost function value of theta0, theta1

t=np.ones((2,1))                                    # temporary theta

for i in range(len(theta0)):                        # storing the costfun values in j_vals
    for j in range(len(theta1)):
        t[0][0]=theta0[i]
        t[1][0]=theta1[j]
        j_vals[i][j]=costfn(t)

plt.contourf(theta0,theta1, j_vals)                 # plot contour
plt.show()

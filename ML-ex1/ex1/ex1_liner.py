import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def computeCost(X,y,theta):
    m=y.size
    cost=0
    cost=sum((np.dot(X,theta)-y)**2)/(2*m)
    return cost
def gradientDescent(X,y,theta,alpha,iterations):
    m=y.size
    J_history=np.zeros((iterations,1))
    for i in range(iterations):
        XT=X.T
        theta=theta-alpha/m*sum(np.dot(XT,np.dot(X,theta)-y))
        J_history[i]=computeCost(X,y,theta)
        return theta,J_history
# ===================== Part 1: Plotting =====================
print('Plotting Data...')
#delimiter是分隔符
data = np.loadtxt('ex1data1.txt', delimiter=',', usecols=(0, 1))
X = data[:, 0]#取第一列人口数作为横坐标
y = data[:, 1]#取第二列利润作为纵坐标
m = y.size    #m=训练样本的个数
X = X.reshape((m, 1))
y = y.reshape((m, 1))
plt.scatter(X,y,alpha=0.6)
plt.xlabel('Population of city in 10,000s')
plt.ylabel('Profit in $10,000s')
input('Program paused. Press ENTER to continue')

# ===================== Part 2: Gradient descent =====================
print('Running Gradient Descent...')

X = np.c_[np.ones(m), X]
theta = np.zeros((2, 1))

iterations = 1500 #迭代次数=1500
alpha = 0.01      #学习率=0.01
print('Initial cost : ' + str(computeCost(X, y, theta)) + ' (This value should be about 32.07)')
theta, J_history = gradientDescent(X, y, theta, alpha, iterations)
print('Theta found by gradient descent: ' + str(theta.reshape(2)))
plt.plot(X[:,1],np.dot(X,theta),'-')
plt.legend(['Linear regression','Training data'])
plt.show()

#画图
line1, = plt.plot(X[:, 1], np.dot(X, theta), label='Linear Regression')
plt.legend(handles=[line1])

input('Program paused. Press ENTER to continue')

# Predict values for population sizes of 35,000 and 70,000
predict1 = np.dot(np.array([1, 3.5]), theta)
print('For population = 35,000, we predict a profit of %f\n'%(predict1[0]*10000))
predict2 = np.dot(np.array([1, 7]), theta)
print('For population = 70,000, we predict a profit of %f\n'%(predict2[0]*10000))
input('Program paused. Press ENTER to continue')

# ===================== Part 3: Visualizing J(theta0, theta1) =====================
print('Visualizing J(theta0, theta1) ...')
theta0_vals = np.linspace(-10, 10, 100)#范围是-10到10，有100个数
theta1_vals = np.linspace(-1, 4, 100)

xs, ys = np.meshgrid(theta0_vals, theta1_vals)
J_vals = np.zeros(xs.shape)

for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        t = np.array([theta0_vals[i], theta1_vals[j]]).reshape((2, 1))
        J_vals[i, j] = computeCost(X, y, t)

J_vals = J_vals.T
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(theta0_vals, theta1_vals, J_vals, rstride=1, cstride=1, cmap='rainbow')
plt.xlabel('theta_0')
plt.ylabel('theta_1')

# 画出等高线
plt.figure()
# 填充颜色，20是等高线分为几部分

plt.contourf(theta0_vals, theta1_vals, J_vals, 20, alpha=0.6, cmap=plt.cm.hot)
plt.contour(theta0_vals, theta1_vals, J_vals, colors='black')
plt.plot(theta[0], theta[1], 'r', marker='x', markerSize=10, LineWidth=2)  # 画点
plt.show()



import csv
import matplotlib.pyplot as plt
import numpy as np

'''
This is a simple implementation of linear regression on correlation
hours studied by student and the marks they obtained.
'''


def run():

	points = np.genfromtxt("data.csv", delimiter=",")
	# x is hours studied, y is marks obtained.
	
	# numpy slicing
	x = points[:,0]
	y = points[:,1]
	# We are applying the function: y = b + mx

	N = len(points)
	b = 0
	m = 0
	alpha = 0.0001 # alpha is the learning rate
	ErrorThreshold  = 0.003
	NumberOfIterations = 3 # We cancel the gradient descent after a number of iterations, if it still doesn't reach the threshold we want.
	
	sum_m = 0
	sum_b = 0

	gradient_list = [sum_m]
	bIntercept_list = [sum_b]

	for i in range(NumberOfIterations):
		if mean_squared_error(x,y,b,m) > ErrorThreshold:
			b , m  = gradient_descent(m,b,alpha,N,x,y)
			gradient_list.append(b)
			bIntercept_list.append(m)
		else:
			break

	linreg_plot(x,y,m,b)


def mean_squared_error(x,y,b,m):
	ErrorValue = 0
	for i in range(len(x)):
		ErrorValue += ((m*x[i] + b) - y[i])**2
	return ErrorValue / len(x) 


def gradient_descent(m,b,alpha,N,x,y):
	
	gradient_b = 0
	gradient_m = 0

	for i in range(len(x)):
		gradient_b += (1/N) * (b + m*x[i] - y[i]) 
		gradient_m += (1/N) * (b + m*x[i] - y[i]) * x[i]
	b , m = b - alpha*gradient_b , m - alpha*gradient_m
	return b,m

def linreg_plot(x,y,m,b):
	plt.xlabel('Hours studied by student')
	plt.ylabel('Marks obtained')
	plt.scatter(x,y,marker='x')
	plt.plot([x for x in range(120)],[m*x + b for x in range(120)])
	plt.show()

if __name__ == '__main__':
	run()
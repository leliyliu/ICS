import matplotlib.pyplot as plt
import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def tanh(x):
    m = np.exp(x)
    n = np.exp(-x)
    return (m-n)/(m+n)

def relu(x):
    return np.maximum(x,0,x)

def leaky_relu(x,a=0.01):
    return np.maximum(a*x,x,x)

def Elu(x,a=1):
    x = np.array([a*(np.exp(t)-1) if t<0 else t for t in x])
    return x

x = np.arange(-10,10,0.01)
plt.title('activation function')


plt.plot(x,sigmoid(x),color = 'skyblue',label = 'sigmoid')
plt.plot(x,tanh(x),color = 'yellow',label = 'tanh')
plt.plot(x,relu(x),color = 'pink',label = 'ReLU')
plt.plot(x,leaky_relu(x),color = 'red',label = 'Leaky ReLU')
plt.plot(x,Elu(x),color = 'blue',label = 'ELU')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.ylim([-2,10])

plt.show()

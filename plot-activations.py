import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.1):
    return np.where(x > 0, x, x * alpha)

def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def swish(x):
    return x * sigmoid(x)

x = np.linspace(-10, 10, 400)

fig, axs = plt.subplots(2, 3, figsize=(15, 10))

# Sigmoid
axs[0, 0].plot(x, sigmoid(x))
axs[0, 0].set_title('Sigmoid: 1 / (1 + exp(-x))')
axs[0, 0].axhline(0, color='black', linewidth=0.5)
axs[0, 0].axvline(0, color='black', linewidth=0.5)

# Tanh
axs[0, 1].plot(x, tanh(x))
axs[0, 1].set_title('Tanh: tanh(x)')
axs[0, 1].axhline(0, color='black', linewidth=0.5)
axs[0, 1].axvline(0, color='black', linewidth=0.5)

# ReLU
axs[0, 2].plot(x, relu(x))
axs[0, 2].set_title('ReLU: max(0, x)')
axs[0, 2].axhline(0, color='black', linewidth=0.5)
axs[0, 2].axvline(0, color='black', linewidth=0.5)

# Leaky ReLU
axs[1, 0].plot(x, leaky_relu(x))
axs[1, 0].set_title('Leaky ReLU: x if x > 0 else 0.01x')
axs[1, 0].axhline(0, color='black', linewidth=0.5)
axs[1, 0].axvline(0, color='black', linewidth=0.5)

# ELU
axs[1, 1].plot(x, elu(x))
axs[1, 1].set_title('ELU: x if x > 0 else alpha*(exp(x)-1)')
axs[1, 1].axhline(0, color='black', linewidth=0.5)
axs[1, 1].axvline(0, color='black', linewidth=0.5)

# Swish
axs[1, 2].plot(x, swish(x))
axs[1, 2].set_title('SiLU: x * sigmoid(x)')
axs[1, 2].axhline(0, color='black', linewidth=0.5)
axs[1, 2].axvline(0, color='black', linewidth=0.5)

plt.tight_layout()
plt.show()

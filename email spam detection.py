import numpy as np
# activation functions
def relu(x):
    return np.maximum(0, x)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
#  weights
W1 = np.array([0.5, -0.2, 0.3])      # Hidden Node 1 weights
W2 = np.array([0.4, 0.1, -0.5])      # Hidden Node 2 weights
Wout = 1                              # Output layer weight

def forward(x):

    # ReLU
    z1 = np.sum(x * W1)
    h1 = relu(z1)
    # Sigmoid
    z2 = h1 * np.sum(W2)
    h2 = sigmoid(z2)
    # Sigmoid
    z3 = h2 * Wout
    output = sigmoid(z3)
    return output



x = np.array([1, 0, 1])
result = forward(x)
print("Final Output:", result)
if result >= 0.5:
    print("Spam Detected: YES (Probability >= 0.5)")
else:
    print("Spam Detected: NO (Probability < 0.5)")


#Final Output: 0.7154518392652967
#Spam Detected: YES (Probability >= 0.5)

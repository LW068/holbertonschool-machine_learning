import numpy as np

class NeuralNetwork:
    def __init__(self, nx, nodes):
        """ Class constructor """
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if not isinstance(nodes, int):
            raise TypeError('nodes must be an integer')
        if nodes < 1:
            raise ValueError('nodes must be a positive integer')

        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """ Weight for hidden layer """
        return self.__W1

    @property
    def b1(self):
        """ Bias for hidden layer """
        return self.__b1

    @property
    def A1(self):
        """ Activated output for hidden layer """
        return self.__A1

    @property
    def W2(self):
        """ Weight for output neuron """
        return self.__W2

    @property
    def b2(self):
        """ Bias for the output neuron """
        return self.__b2

    @property
    def A2(self):
        """ Activated output for the output neuron """
        return self.__A2

    def forward_prop(self, X):
        """ Calculates the forward propagation """
        Z1 = np.matmul(self.W1, X) + self.b1
        self.__A1 = 1 / (1 + np.exp(-Z1))
        Z2 = np.matmul(self.W2, self.A1) + self.b2
        self.__A2 = 1 / (1 + np.exp(-Z2))

    def cost(self, Y, A):
        """ Calculates the cost """
        m = Y.shape[1]
        cost = -1/m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """ Evaluates the neurons predictions """
        self.forward_prop(X)
        cost = self.cost(Y, self.A2)
        return np.where(self.A2 >= 0.5, 1, 0), cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """ Calculates the gradient descent """
        m = Y.shape[1]
        dZ2 = A2 - Y
        dW2 = np.matmul(dZ2, A1.T) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m
        dZ1 = np.matmul(self.W2.T, dZ2) * (A1 * (1 - A1))
        dW1 = np.matmul(dZ1, X.T) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m

        self.__W1 = self.W1 - alpha * dW1
        self.__b1 = self.b1 - alpha * db1
        self.__W2 = self.W2 - alpha * dW2
        self.__b2 = self.b2 - alpha * db2

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """ Trains the neural network """
        if not isinstance(iterations, int):
            raise TypeError('iterations must be an integer')
        if iterations <= 0:
            raise ValueError('iterations must be a positive integer')

        if not isinstance(alpha, float):
            raise TypeError('alpha must be a float')
        if alpha <= 0:
            raise ValueError('alpha must be positive')

        for _ in range(iterations):
            self.forward_prop(X)
            self.gradient_descent(X, Y, self.A1, self.A2, alpha)

        return self.evaluate(X, Y)

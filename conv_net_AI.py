import numpy as np
import pandas as pd

# Load data
data = pd.read_csv('data/train.csv')
labels = data['label'].values
images = data.drop('label', axis=1).values

# Normalize the images
images = images / 255.0
images = images.reshape(-1, 28, 28, 1)  # Reshape to (num_samples, height, width, channels)

# One-hot encode the labels
num_classes = 10
labels = np.eye(num_classes)[labels]

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def relu_derivative(x):
    return (x > 0).astype(float)

def cross_entropy_loss(y_pred, y_true):
    return -np.mean(np.sum(y_true * np.log(y_pred + 1e-9), axis=1))

def accuracy(y_pred, y_true):
    return np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1))

class Conv2D:
    def __init__(self, num_filters, filter_size, input_shape):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.input_shape = input_shape
        self.filters = np.random.randn(num_filters, filter_size, filter_size, input_shape[3]) / filter_size ** 2

    def forward(self, input):
        self.input = input
        self.h, self.w, self.c = input.shape[1:]
        self.output = np.zeros((input.shape[0], self.h - self.filter_size + 1, self.w - self.filter_size + 1, self.num_filters))
        for i in range(self.h - self.filter_size + 1):
            for j in range(self.w - self.filter_size + 1):
                region = input[:, i:i+self.filter_size, j:j+self.filter_size, :]
                self.output[:, i, j, :] = np.tensordot(region, self.filters, axes=([1, 2, 3], [1, 2, 3]))
        return self.output

    def backward(self, d_output, learning_rate):
        d_filters = np.zeros(self.filters.shape)
        for i in range(self.h - self.filter_size + 1):
            for j in range(self.w - self.filter_size + 1):
                region = self.input[:, i:i+self.filter_size, j:j+self.filter_size, :]
                for k in range(self.num_filters):
                    d_filters[k] += np.sum(region * d_output[:, i, j, k][:, None, None, None], axis=0)
        self.filters -= learning_rate * d_filters / self.input.shape[0]


class MaxPool2D:
    def __init__(self, pool_size):
        self.pool_size = pool_size

    def forward(self, input):
        self.input = input
        self.h, self.w, self.c = input.shape[1:]
        self.output = np.zeros((input.shape[0], self.h // self.pool_size, self.w // self.pool_size, self.c))
        for i in range(0, self.h, self.pool_size):
            for j in range(0, self.w, self.pool_size):
                self.output[:, i//self.pool_size, j//self.pool_size, :] = np.max(input[:, i:i+self.pool_size, j:j+self.pool_size, :], axis=(1, 2))
        return self.output

    def backward(self, d_output):
        d_input = np.zeros(self.input.shape)
        for i in range(0, self.h, self.pool_size):
            for j in range(0, self.w, self.pool_size):
                region = self.input[:, i:i+self.pool_size, j:j+self.pool_size, :]
                max_region = np.max(region, axis=(1, 2), keepdims=True)
                d_input[:, i:i+self.pool_size, j:j+self.pool_size, :] = (region == max_region) * d_output[:, i//self.pool_size, j//self.pool_size, :][:, None, None, :]
        return d_input

class FullyConnected:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) / input_size
        self.biases = np.zeros(output_size)

    def forward(self, input):
        self.input = input
        return np.dot(input, self.weights) + self.biases

    def backward(self, d_output, learning_rate):
        d_input = np.dot(d_output, self.weights.T)
        d_weights = np.dot(self.input.T, d_output)
        d_biases = np.sum(d_output, axis=0)
        self.weights -= learning_rate * d_weights / self.input.shape[0]
        self.biases -= learning_rate * d_biases / self.input.shape[0]
        return d_input

conv = Conv2D(8, 3, images.shape)
pool = MaxPool2D(2)
fc = FullyConnected(13 * 13 * 8, num_classes)

learning_rate = 0.01
epochs = 10
batch_size = 32

for epoch in range(epochs):
    indices = np.arange(images.shape[0])
    np.random.shuffle(indices)
    for start in range(0, images.shape[0], batch_size):
        end = start + batch_size
        x_batch = images[indices[start:end]]
        y_batch = labels[indices[start:end]]

        # Forward pass
        conv_output = conv.forward(x_batch)
        pool_output = pool.forward(conv_output)
        pool_output_flat = pool_output.reshape(pool_output.shape[0], -1)
        logits = fc.forward(pool_output_flat)
        y_pred = softmax(logits)

        # Compute loss and accuracy
        loss = cross_entropy_loss(y_pred, y_batch)
        acc = accuracy(y_pred, y_batch)

        # Backward pass
        d_logits = y_pred - y_batch
        d_fc = fc.backward(d_logits, learning_rate)
        d_pool_flat = d_fc.reshape(pool_output.shape)
        d_pool = pool.backward(d_pool_flat)
        d_conv = conv.backward(d_pool, learning_rate)
        print(f'Epoch {epoch + 1}, Loss: {loss:.4f}, Accuracy: {acc:.4f}')

    print(f'Epoch {epoch + 1}, Loss: {loss:.4f}, Accuracy: {acc:.4f}')

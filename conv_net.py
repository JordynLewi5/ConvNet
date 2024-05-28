import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

class ConvNet:
    @staticmethod
    def init_parameters():
        layers_data = {
            "conv1": {
                "filters": [np.random.rand(3, 3) for _ in range(10)],
                "bias": np.zeros(10),
                "padding_width": 1,
                "padding_height": 1,
                "stride_width": 1,
                "stride_height": 1
            },
            "pool1": {
                "pool_width": 2,
                "pool_height": 2,
                "mode": "max"
            },
            "conv2": {
                "filters": [np.random.rand(3, 3) for _ in range(10)],
                "bias": np.zeros(10),
                "padding_width": 1,
                "padding_height": 1,
                "stride_width": 1,
                "stride_height": 1
            },
            "pool2": {
                "pool_width": 2,
                "pool_height": 2,
                "mode": "max"
            },
            "batch_norm1": {
                "gamma": np.ones(10),
                "beta": np.zeros(10),
                "mean": np.zeros(10),
                "variance": np.ones(10)
            },
            "batch_norm2": {
                "gamma": np.ones(10),
                "beta": np.zeros(10),
                "mean": np.zeros(10),
                "variance": np.ones(10)
            }
        }
        return layers_data

    @staticmethod
    def relu2d(matrix):
        return np.maximum(matrix, 0)

    @staticmethod
    def relu2d_derivative(matrix):
        return np.where(matrix > 0, 1, 0)

    @staticmethod
    def sigmoid2d(matrix):
        return 1 / (1 + np.exp(-matrix))

    @staticmethod
    def apply_relu(feature_maps):
        return np.array([ConvNet.relu2d(feature_map) for feature_map in feature_maps])

    @staticmethod
    def apply_relu_derivative(feature_maps):
        return np.array([ConvNet.relu2d_derivative(feature_map) for feature_map in feature_maps])

    @staticmethod
    def cross_entropy_loss(predictions, targets):
        predictions = np.clip(predictions, 1e-12, 1 - 1e-12)
        return -np.sum(targets * np.log(predictions))

    @staticmethod
    def softmax(z):
        shift_z = z - np.max(z)
        exp_z = np.exp(shift_z)
        return exp_z / exp_z.sum(axis=0, keepdims=True)


class ConvLayer:
    def __init__(self, filters, bias, padding_width, padding_height, stride_width, stride_height):
        self.filters = filters
        self.bias = bias
        self.padding_width = padding_width
        self.padding_height = padding_height
        self.stride_width = stride_width
        self.stride_height = stride_height

    def convolution2d(self, matrix, filter, padding_width=1, padding_height=1, stride_width=1, stride_height=1):
        input_height, input_width = matrix.shape
        filter_height, filter_width = filter.shape
        output_width = int((input_width + 2 * padding_width - filter_width) / stride_width + 1)
        output_height = int((input_height + 2 * padding_height - filter_height) / stride_height + 1)
        output = np.zeros((output_height, output_width))

        padded_matrix = np.pad(matrix, ((padding_height, padding_height), (padding_width, padding_width)), mode="constant")

        for row_index in range(0, output_height):
            for column_index in range(0, output_width):
                pixel_sum = 0
                for filter_row_index, filter_row in enumerate(filter):
                    for filter_column_index, filter_weight in enumerate(filter_row):
                        pixel_sum += filter_weight * padded_matrix[row_index * stride_height + filter_row_index][column_index * stride_width + filter_column_index]
                output[row_index][column_index] = pixel_sum

        return output

    def forward(self, feature_maps):
        num_filters = len(self.filters)
        output_feature_maps = []
        for i in range(num_filters):
            filter = self.filters[i]
            b = self.bias[i]
            filtered_maps = [self.convolution2d(feature_map, filter, self.padding_width, self.padding_height, self.stride_width, self.stride_height) for feature_map in feature_maps]
            combined_map = np.sum(filtered_maps, axis=0) + b
            output_feature_maps.append(combined_map)
        return np.array(output_feature_maps)

    def backward_conv(self, d_out, input_matrix):
        num_filters = len(self.filters)
        d_filters = [np.zeros(filter.shape) for filter in self.filters]
        d_bias = np.zeros(self.bias.shape)
        d_input = np.zeros(input_matrix.shape)

        padded_input = np.pad(input_matrix, ((0, 0), (self.padding_height, self.padding_height), (self.padding_width, self.padding_width)), mode="constant")
        d_padded_input = np.zeros(padded_input.shape)

        for i in range(num_filters):
            d_bias[i] = np.sum(d_out[i])
            for j in range(input_matrix.shape[0]):  # Loop over each input feature map
                for row in range(d_out.shape[1]):
                    for col in range(d_out.shape[2]):
                        region = padded_input[j, row * self.stride_height: row * self.stride_height + self.filters[i].shape[0],
                                              col * self.stride_width: col * self.stride_width + self.filters[i].shape[1]]
                        d_filters[i] += d_out[i, row, col] * region
                        d_padded_input[j, row * self.stride_height: row * self.stride_height + self.filters[i].shape[0],
                                       col * self.stride_width: col * self.stride_width + self.filters[i].shape[1]] += d_out[i, row, col] * self.filters[i]

        # Remove padding from d_padded_input to get d_input
        if self.padding_height > 0 and self.padding_width > 0:
            d_input = d_padded_input[:, self.padding_height:-self.padding_height, self.padding_width:-self.padding_width]
        else:
            d_input = d_padded_input

        return d_input, np.array(d_filters), d_bias

class PoolingLayer:
    def __init__(self, pool_width, pool_height, mode):
        self.pool_width = pool_width
        self.pool_height = pool_height
        self.mode = mode
        
    @staticmethod
    def maxpooling2d(matrix, pool_width, pool_height):
        input_height, input_width = matrix.shape
        output_width = int(input_width / pool_width)
        output_height = int(input_height / pool_height)

        output = np.zeros((output_height, output_width))

        for row_index in range(0, output_height):
            for column_index in range(0, output_width):
                pooled_pixels_values = []
                for pooling_row_index in range(0, pool_height):
                    for pooling_column_index in range(0, pool_width):
                        pooled_pixels_values.append(matrix[row_index * pool_height + pooling_row_index][column_index * pool_width + pooling_column_index])
                
                output[row_index][column_index] = np.max(pooled_pixels_values)

        return output
    
    @staticmethod
    def averagepooling2d(matrix, pool_width, pool_height):
        input_height, input_width = matrix.shape
        output_width = int(input_width / pool_width)
        output_height = int(input_height / pool_height)

        output = np.zeros((output_height, output_width))

        for row_index in range(0, output_height):
            for column_index in range(0, output_width):
                pooled_pixels_values = []
                for pooling_row_index in range(0, pool_height):
                    for pooling_column_index in range(0, pool_width):
                        pooled_pixels_values.append(matrix[row_index * pool_height + pooling_row_index][column_index * pool_width + pooling_column_index])
                
                output[row_index][column_index] = np.mean(pooled_pixels_values)

        return output
    
    def forward(self, feature_maps):
        if self.mode == "max":
            return np.array([self.maxpooling2d(feature_map, self.pool_width, self.pool_height) for feature_map in feature_maps])
        elif self.mode == "average":
            return np.array([self.averagepooling2d(feature_map, self.pool_width, self.pool_height) for feature_map in feature_maps])
        
    def backward_pool(self, d_out, input_matrix):
        d_input = np.zeros(input_matrix.shape)
        pool_height, pool_width = self.pool_height, self.pool_width
        
        for i in range(d_out.shape[0]):
            for row in range(d_out.shape[1]):
                for col in range(d_out.shape[2]):
                    row_start = row * pool_height
                    col_start = col * pool_width
                    if self.mode == "max":
                        window = input_matrix[i, row_start:row_start+pool_height, col_start:col_start+pool_width]
                        max_val = np.max(window)
                        for row_in_pool in range(pool_height):
                            for col_in_pool in range(pool_width):
                                if window[row_in_pool, col_in_pool] == max_val:
                                    d_input[i, row_start+row_in_pool, col_start+col_in_pool] = d_out[i, row, col]
                    elif self.mode == "average":
                        average_grad = d_out[i, row, col] / (pool_height * pool_width)
                        for row_in_pool in range(pool_height):
                            for col_in_pool in range(pool_width):
                                d_input[i, row_start+row_in_pool, col_start+col_in_pool] += average_grad
        
        return d_input

class FullyConnectedLayer:
    def __init__(self, input_size, output_size):
        if input_size == 0:
            raise ValueError("Input size for fully connected layer must be greater than zero.")
        self.weights = np.random.randn(output_size, input_size) * np.sqrt(2 / input_size) - 0.5
        self.bias = np.random.randn(output_size, 1) * np.sqrt(2 / input_size) - 0.5

    def forward(self, input_vector):
        return self.weights.dot(input_vector) + self.bias

    def backward_fc(self, d_out, input_vector):
        d_weights = d_out.dot(input_vector.T)
        d_bias = d_out
        d_input = self.weights.T.dot(d_out)
        return d_input, d_weights, d_bias


class BatchNormLayer:
    def __init__(self, gamma, beta, mean, variance, epsilon=1e-5):
        self.gamma = gamma
        self.beta = beta
        self.mean = mean
        self.variance = variance
        self.epsilon = epsilon

    def forward(self, feature_maps):
        self.mean = np.mean(feature_maps, axis=(1, 2))
        self.variance = np.var(feature_maps, axis=(1, 2))
        self.normalized = (feature_maps - self.mean[:, None, None]) / np.sqrt(self.variance[:, None, None] + self.epsilon)
        return self.gamma[:, None, None] * self.normalized + self.beta[:, None, None]

    def backward(self, d_out, feature_maps):
        N, H, W = feature_maps.shape
        d_normalized = d_out * self.gamma[:, None, None]
        d_variance = np.sum(d_normalized * (feature_maps - self.mean[:, None, None]) * -0.5 * (self.variance[:, None, None] + self.epsilon)**-1.5, axis=(1, 2))
        d_mean = np.sum(d_normalized * -1 / np.sqrt(self.variance[:, None, None] + self.epsilon), axis=(1, 2)) + d_variance * np.sum(-2 * (feature_maps - self.mean[:, None, None]), axis=(1, 2)) / (H * W)
        d_input = d_normalized / np.sqrt(self.variance[:, None, None] + self.epsilon) + d_variance[:, None, None] * 2 * (feature_maps - self.mean[:, None, None]) / (H * W) + d_mean[:, None, None] / (H * W)
        d_gamma = np.sum(d_out * self.normalized, axis=(1, 2))
        d_beta = np.sum(d_out, axis=(1, 2))
        return d_input, d_gamma, d_beta


def plot_charts(matrices):
    fig, axes = plt.subplots(1, len(matrices), figsize=(20, 5))
    if len(matrices) == 1:
        axes = [axes]
    for i, matrix in enumerate(matrices):
        axes[i].imshow(matrix, cmap="viridis")
        axes[i].set_title(f"Matrix {i+1}")
        axes[i].axis("off")
    plt.show()

def clip_gradients(gradients, threshold=1.0):
    clipped_gradients = []
    for gradient in gradients:
        np.clip(gradient, -threshold, threshold, out=gradient)
        clipped_gradients.append(gradient)
    return clipped_gradients

######################################################

def backward_pass(softmax_output, label, flattened_input, pooled_maps2, feature_maps2, pooled_maps1, feature_maps1, input_matrix, conv1, pool1, conv2, pool2, fc_layer, batch_norm1, batch_norm2):
    # One-hot encode the label
    label_one_hot = np.zeros(softmax_output.shape)
    label_one_hot[label] = 1
    
    # Compute gradient of loss with respect to softmax output
    dL_dsoftmax = softmax_output - label_one_hot

    # Backpropagate through fully connected layer
    dL_dfc_input, dL_dfc_weights, dL_dfc_bias = fc_layer.backward_fc(dL_dsoftmax, flattened_input)

    # Clip gradients to prevent overflow
    dL_dfc_weights, dL_dfc_bias = clip_gradients([dL_dfc_weights, dL_dfc_bias])

    # Reshape the gradient to match the pooled map shape
    dL_dpooled_maps2 = dL_dfc_input.reshape(pooled_maps2.shape)
    
    # Backpropagate through second pooling layer
    dL_dfeature_maps2 = pool2.backward_pool(dL_dpooled_maps2, feature_maps2)

    # Apply ReLU derivative
    dL_dfeature_maps2 = dL_dfeature_maps2 * ConvNet.apply_relu_derivative(feature_maps2)
    
    # Backpropagate through second batch normalization layer
    dL_dfeature_maps2, dL_dgamma2, dL_dbeta2 = batch_norm2.backward(dL_dfeature_maps2, feature_maps2)
    
    # Backpropagate through second convolutional layer
    dL_dinput_pooled_maps1, dL_dconv2_filters, dL_dconv2_bias = conv2.backward_conv(dL_dfeature_maps2, pooled_maps1)
    
    # Clip gradients to prevent overflow
    dL_dconv2_filters, dL_dconv2_bias = clip_gradients([dL_dconv2_filters, dL_dconv2_bias])
    
    # Backpropagate through first pooling layer
    dL_dfeature_maps1 = pool1.backward_pool(dL_dinput_pooled_maps1, feature_maps1)
    
    # Apply ReLU derivative
    dL_dfeature_maps1 = dL_dfeature_maps1 * ConvNet.apply_relu_derivative(feature_maps1)
    
    # Backpropagate through first batch normalization layer
    dL_dfeature_maps1, dL_dgamma1, dL_dbeta1 = batch_norm1.backward(dL_dfeature_maps1, feature_maps1)
    
    # Backpropagate through first convolutional layer
    dL_dinput_input_matrix, dL_dconv1_filters, dL_dconv1_bias = conv1.backward_conv(dL_dfeature_maps1, np.array([input_matrix]))
    
    # Clip gradients to prevent overflow
    dL_dconv1_filters, dL_dconv1_bias = clip_gradients([dL_dconv1_filters, dL_dconv1_bias])
    
    return dL_dfc_weights, dL_dfc_bias, dL_dconv2_filters, dL_dconv2_bias, dL_dconv1_filters, dL_dconv1_bias, dL_dgamma1, dL_dbeta1, dL_dgamma2, dL_dbeta2

def update_parameters(fc_layer, layers_data, dL_dfc_weights, dL_dfc_bias, dL_dconv2_filters, dL_dconv2_bias, dL_dconv1_filters, dL_dconv1_bias, dL_dgamma1, dL_dbeta1, dL_dgamma2, dL_dbeta2, learning_rate):    
    # Update fully connected layer parameters
    fc_layer.weights -= learning_rate * dL_dfc_weights
    fc_layer.bias -= learning_rate * dL_dfc_bias

    # Update second convolutional layer parameters
    for i in range(len(layers_data["conv2"]["filters"])):
        layers_data["conv2"]["filters"][i] -= learning_rate * dL_dconv2_filters[i]
        layers_data["conv2"]["bias"][i] -= learning_rate * dL_dconv2_bias[i]
    
    # Update first convolutional layer parameters
    for i in range(len(layers_data["conv1"]["filters"])):
        layers_data["conv1"]["filters"][i] -= learning_rate * dL_dconv1_filters[i]
        layers_data["conv1"]["bias"][i] -= learning_rate * dL_dconv1_bias[i]

    # Update batch normalization parameters
    layers_data["batch_norm1"]["gamma"] -= learning_rate * dL_dgamma1
    layers_data["batch_norm1"]["beta"] -= learning_rate * dL_dbeta1
    layers_data["batch_norm2"]["gamma"] -= learning_rate * dL_dgamma2
    layers_data["batch_norm2"]["beta"] -= learning_rate * dL_dbeta2

    return layers_data

def forward_pass(input_matrix, layers_data, label):
    conv1 = ConvLayer(
        layers_data["conv1"]["filters"],
        layers_data["conv1"]["bias"],
        layers_data["conv1"]["padding_width"],
        layers_data["conv1"]["padding_height"],
        layers_data["conv1"]["stride_width"],
        layers_data["conv1"]["stride_height"]
    )

    feature_maps1 = conv1.forward(np.array([input_matrix]))
    feature_maps1 = ConvNet.apply_relu(feature_maps1)
    # print("Feature maps 1")
    # print(feature_maps1)

    batch_norm1 = BatchNormLayer(
        layers_data["batch_norm1"]["gamma"],
        layers_data["batch_norm1"]["beta"],
        layers_data["batch_norm1"]["mean"],
        layers_data["batch_norm1"]["variance"]
    )
    feature_maps1 = batch_norm1.forward(feature_maps1)

    pool1 = PoolingLayer(
        layers_data["pool1"]["pool_width"],
        layers_data["pool1"]["pool_height"],
        layers_data["pool1"]["mode"]
    )
    pooled_maps1 = pool1.forward(feature_maps1)

    conv2 = ConvLayer(
        layers_data["conv2"]["filters"],
        layers_data["conv2"]["bias"],
        layers_data["conv2"]["padding_width"],
        layers_data["conv2"]["padding_height"],
        layers_data["conv2"]["stride_width"],
        layers_data["conv2"]["stride_height"]
    )

    feature_maps2 = conv2.forward(pooled_maps1)
    feature_maps2 = ConvNet.apply_relu(feature_maps2)
    # print("Feature maps 2")
    # for feature_map in feature_maps2:
    #     print(str(feature_map)) 

    batch_norm2 = BatchNormLayer(
        layers_data["batch_norm2"]["gamma"],
        layers_data["batch_norm2"]["beta"],
        layers_data["batch_norm2"]["mean"],
        layers_data["batch_norm2"]["variance"]
    )
    feature_maps2 = batch_norm2.forward(feature_maps2)

    pool2 = PoolingLayer(
        layers_data["pool2"]["pool_width"],
        layers_data["pool2"]["pool_height"],
        layers_data["pool2"]["mode"]
    )
    pooled_maps2 = pool2.forward(feature_maps2)

    flattened_input = pooled_maps2.flatten().reshape(-1, 1)
    # print("Flattened input")
    # print(flattened_input)

    fc_layer = FullyConnectedLayer(flattened_input.size, 10)
    output1 = fc_layer.forward(flattened_input)

    softmax_output = ConvNet.softmax(output1)
    # print("Softmax output")
    # print(softmax_output)

    return softmax_output, label, flattened_input, pooled_maps2, feature_maps2, pooled_maps1, feature_maps1, input_matrix, conv1, pool1, conv2, pool2, fc_layer, batch_norm1, batch_norm2

def train(data, layers_data, epochs, initial_learning_rate):
    for epoch in range(epochs):
        np.random.shuffle(data)  # Shuffle the data for each epoch
        total_loss = 0
        accuracy = 0
        learning_rate = initial_learning_rate * (0.95 ** epoch)  # Learning rate scheduling
        print("Learning rate:", learning_rate)
        for i in range(len(data)):
            input_matrix = data[i][1:].reshape(28, 28)
            label = int(data[i][0])

            # Forward pass
            softmax_output, label, flattened_input, pooled_maps2, feature_maps2, pooled_maps1, feature_maps1, input_matrix, conv1, pool1, conv2, pool2, fc_layer, batch_norm1, batch_norm2 = forward_pass(
                input_matrix, layers_data, label)

            # One-hot encode the label
            label_one_hot = np.zeros(softmax_output.shape)
            label_one_hot[label] = 1

            loss = ConvNet.cross_entropy_loss(softmax_output, label_one_hot)
            total_loss += loss

            # Compute accuracy
            predictions = np.argmax(softmax_output, axis=0)
            accuracy += int(predictions == label)

            # Backpropagation
            dL_dfc_weights, dL_dfc_bias, dL_dconv2_filters, dL_dconv2_bias, dL_dconv1_filters, dL_dconv1_bias, dL_dgamma1, dL_dbeta1, dL_dgamma2, dL_dbeta2 = backward_pass(
                softmax_output, label, flattened_input, pooled_maps2, feature_maps2, pooled_maps1, feature_maps1, input_matrix, conv1, pool1, conv2, pool2, fc_layer, batch_norm1, batch_norm2)

            # Update parameters
            layers_data = update_parameters(fc_layer, layers_data, dL_dfc_weights, dL_dfc_bias, dL_dconv2_filters, dL_dconv2_bias, dL_dconv1_filters, dL_dconv1_bias, dL_dgamma1, dL_dbeta1, dL_dgamma2, dL_dbeta2, learning_rate)

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(data)}, Accuracy: {accuracy / len(data)}")

    return layers_data


######################################################

# Load and preprocess the data
data = pd.read_csv('./data/train.csv')
data = np.array(data, dtype=float)  # Ensure data is float before division
data[:, 1:] = data[:, 1:] / 255.0
data = data[:10]  # Use only the first 100 samples for faster training

# Initialize the parameters
layers_data = ConvNet.init_parameters()

# Train the model
epochs = 10
initial_learning_rate = 0.1
trained_layers_data = train(data, layers_data, epochs, initial_learning_rate)

# Save the trained model
np.save('data/trained_layers_data.npy', trained_layers_data)

# Test the model on a sample image
data = pd.read_csv('./data/train.csv')
data = np.array(data, dtype=float)
data[:, 1:] = data[:, 1:] / 255.0
test_image = data[1][1:].reshape(28, 28)
test_label = data[1][0]

# Forward pass through the model
softmax_output, label, flattened_input, pooled_maps2, feature_maps2, pooled_maps1, feature_maps1, input_matrix, conv1, pool1, conv2, pool2, fc_layer, batch_norm1, batch_norm2 = forward_pass(
    test_image, trained_layers_data, test_label)

# Make predictions
predictions = np.argmax(softmax_output, axis=0)
print("Predicted Label:", predictions)
print("True Label:", test_label)

# Plot the charts (if required)
plot_charts([test_image] + [feature_map for feature_map in feature_maps1] + [pooled_map for pooled_map in pooled_maps1] + [feature_map for feature_map in feature_maps2] + [pooled_map for pooled_map in pooled_maps2] + [flattened_input])

import numpy as np
from torchvision.datasets import MNIST
from sklearn.preprocessing import OneHotEncoder
import time 
from PIL import Image

# 1 - Load the MNIST dataset 
def download_mnist(is_train: bool):
    dataset = MNIST(root='./data',
                    transform=lambda x: np.array(x) .flatten(),
                    download=True,
                    train=is_train)

    mnist_data = []
    mnist_labels = []
    for image, label in dataset:
        mnist_data. append(image)
        mnist_labels. append(label)

    mnist_data = np.array(mnist_data)
    mnist_labels = np.array(mnist_labels)
    return mnist_data, mnist_labels

    return mnist_data, mnist_labels

train_X, train_Y = download_mnist(True)
test_X, test_Y = download_mnist(False)


# 2 - Normalize the data and convert the labels to one-hot-encoding.
# Normalize the data
train_X = train_X / 255.0
test_X = test_X / 255.0

# Convert the labels to one-hot-encoding
encoder = OneHotEncoder(categories='auto', sparse_output=False)
train_Y = encoder.fit_transform(train_Y.reshape(-1, 1))
test_Y = encoder.transform(test_Y.reshape(-1, 1))


# 3 - Initialize Weights and Biases
np.random.seed(42)  # For reproducibility

input_size = 784  # 28x28 images
output_size = 10   # output classes (0-9 digits)

W = np.random.randn(input_size, output_size) * 0.01 
b = np.zeros((output_size,))


# 4 - Forward Propagation Using Softmax
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def forward_propagation(X, W, b):
    Z = np.dot(X, W) + b  
    A = softmax(Z)       
    return A

def compute_loss(Y, A):
    m = Y.shape[0] 
    loss = -np.sum(Y * np.log(A + 1e-8)) / m 
    return loss

# 5 - Backward Propagation Using Gradient Descent
def backward_propagation(X, Y, A, W, b, learning_rate=0.01):
    m = X.shape[0]  # Batch size
    
    dZ = A - Y  # eroarea (predictii - etichete reale)
    dW = np.dot(X.T, dZ) / m  # gradientul ponderilor
    db = np.sum(dZ, axis=0) / m  # gradientul bias-urilor
    
    # gradient descent
    W -= learning_rate * dW
    b -= learning_rate * db
    
    return W, b

def train(X_train, Y_train, X_test, Y_test, W, b, epochs, batch_size, learning_rate):
    m = X_train.shape[0]

    start_time = time.time()

    for epoch in range(epochs):
        permutation = np.random.permutation(m)
        X_train_shuffled = X_train[permutation]
        Y_train_shuffled = Y_train[permutation]

        for i in range(0, m, batch_size):
            X_batch = X_train_shuffled[i:i+batch_size]
            Y_batch = Y_train_shuffled[i:i+batch_size]

            A = forward_propagation(X_batch, W, b)

            W, b = backward_propagation(X_batch, Y_batch, A, W, b, learning_rate)

        # training accuracy
        A_train = forward_propagation(X_train, W, b)
        predictions_train = np.argmax(A_train, axis=1)
        labels_train = np.argmax(Y_train, axis=1)
        accuracy_train = np.mean(predictions_train == labels_train)

        # test accuracy
        A_test = forward_propagation(X_test, W, b)
        predictions_test = np.argmax(A_test, axis=1)
        labels_test = np.argmax(Y_test, axis=1)
        accuracy_test = np.mean(predictions_test == labels_test)

        print(f"Epoch {epoch + 1}/{epochs} - Training Accuracy: {accuracy_train:.4f} - Test Accuracy: {accuracy_test:.4f}")
    
    end_time = time.time()  
    total_time = end_time - start_time  
    print(f"Total training time: {total_time:.2f} seconds")

    return W, b

W, b = train(train_X, train_Y, test_X, test_Y, W, b, epochs=200, batch_size=100, learning_rate=0.01)


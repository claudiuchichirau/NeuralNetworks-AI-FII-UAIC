import numpy as np
from torchvision.datasets import MNIST
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
import time 
from PIL import Image

# 1 - MNIST dataset 
def download_mnist(is_train: bool):
    dataset = MNIST(root='./data',
                    transform=lambda x: np.array(x) .flatten(),
                    download=True,
                    train=is_train)

    mnist_data = []
    mnist_labels = []
    for image, label in dataset:
        mnist_data.append(image)
        mnist_labels.append(label)

    mnist_data = np.array(mnist_data)
    mnist_labels = np.array(mnist_labels)
    return mnist_data, mnist_labels

    return mnist_data, mnist_labels

train_X, train_Y = download_mnist(True)
test_X, test_Y = download_mnist(False)


# Normalize the data
train_X = train_X / 255.0
test_X = test_X / 255.0

# Convert the labels to one-hot-encoding
encoder = OneHotEncoder(categories='auto', sparse_output=False)
train_Y = encoder.fit_transform(train_Y.reshape(-1, 1))
test_Y = encoder.transform(test_Y.reshape(-1, 1))


# 3 - Forward Propagation Using Softmax and Relu Activation Function 
def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return z > 0

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def initialize_parameters(layer_dims):
    parameters = {}
    for l in range(1, len(layer_dims)):
        parameters[f'W{l}'] = np.random.randn(layer_dims[l-1], layer_dims[l]) * np.sqrt(2. / layer_dims[l-1])  # He initialization
        # parameters[f'W{l}'] = np.random.randn(layer_dims[l-1], layer_dims[l]) * 0.01
        parameters[f'b{l}'] = np.zeros((1, layer_dims[l]))
    return parameters

def apply_dropout(A, dropout_rate):
    D = np.random.rand(A.shape[0], A.shape[1]) < dropout_rate
    A *= D 
    A /= dropout_rate  
    return A, D

def forward_propagation(X, parameters, num_layers, dropout_rate):
    caches = []
    A = X 
    
    for layer in range(1, num_layers):
        W = parameters[f'W{layer}']
        b = parameters[f'b{layer}']
        Z = np.dot(A, W) + b
        if layer < num_layers - 1:
            A = relu(Z)     # ReLU for hidden layers
            A, D = apply_dropout(A, dropout_rate)
            caches.append((A, Z, D)) 
        else:
            A = softmax(Z)  # Softmax for output
            caches.append((A, Z))
    
    return A, caches

def compute_loss(Y, A):
    m = Y.shape[0] 
    loss = -np.sum(Y * np.log(A + 1e-8)) / m 
    return loss


def backward_propagation(X, Y, caches, parameters, learning_rate, num_layers, dropout_rate, lambda_l1):
    m = X.shape[0] 
    grads = {}

    A_last, Z_last = caches[-1]
    dZ_last = A_last - Y  # loss function derivative for the last layer
    
    # weights & bias gradients for the last layer
    A_prev, Z_prev = caches[-2][:2]
    dW_last = np.dot(A_prev.T, dZ_last) / m + (lambda_l1 / m) * np.sign(parameters[f'W{num_layers-1}']) # L1 regularization
    db_last = np.sum(dZ_last, axis=0, keepdims=True) / m
    grads[f'dW{num_layers-1}'] = dW_last
    grads[f'db{num_layers-1}'] = db_last
    
    # propagate error back through hidden layers
    dA = np.dot(dZ_last, parameters[f'W{num_layers-1}'].T)

    for layer in reversed(range(1, num_layers-1)):
        if layer > 1:
            A_prev = caches[layer-1][0]
            D = caches[layer-1][2]  
            dA *= D
            dA /= dropout_rate
        else:   
            A_prev = X       

        Z = caches[layer-1][1] 

        # propagate error through activated neurons 
        dZ = dA * relu_derivative(Z)
        
        dW = np.dot(A_prev.T, dZ) / m + (lambda_l1 / m) * np.sign(parameters[f'W{layer}'])  # L1 regularization
        db = np.sum(dZ, axis=0, keepdims=True) / m
        
        grads[f'dW{layer}'] = dW
        grads[f'db{layer}'] = db
        
        dA = np.dot(dZ, parameters[f'W{layer}'].T)
    
    for l in range(1, num_layers):
        parameters[f'W{layer}'] -= learning_rate * grads[f'dW{layer}']
        parameters[f'b{layer}'] -= learning_rate * grads[f'db{layer}']
    
    return parameters


def train(train_X, train_Y, test_X, test_Y, layer_dims, epochs, batch_size, initial_learning_rate, dropout_rate, lambda_l1):
    parameters = initialize_parameters(layer_dims)
    num_layers = len(layer_dims)
    
    start_time = time.time()

    for epoch in range(epochs):
        permutation = np.random.permutation(train_X.shape[0])
        train_X_shuffled = train_X[permutation]
        train_Y_shuffled = train_Y[permutation]
        
        epoch_loss = 0  
        num_batches = 0

        for i in range(0, train_X.shape[0], batch_size):
            X_batch = train_X_shuffled[i:i+batch_size]
            Y_batch = train_Y_shuffled[i:i+batch_size]
        
            A, caches = forward_propagation(X_batch, parameters, num_layers, dropout_rate)
            
            loss = compute_loss(Y_batch, A)
            epoch_loss += loss 
            num_batches += 1  

            parameters = backward_propagation(X_batch, Y_batch, caches, parameters, initial_learning_rate, num_layers, dropout_rate, lambda_l1)
        
        average_loss = epoch_loss / num_batches

        # training accuracy 
        A_train, _ = forward_propagation(train_X, parameters, num_layers, dropout_rate)
        predictions_train = np.argmax(A_train, axis=1)
        labels_train = np.argmax(train_Y, axis=1)
        accuractrain_Y = np.mean(predictions_train == labels_train)

        # test accuracy
        A_test, _ = forward_propagation(test_X, parameters, num_layers, dropout_rate)
        predictions_test = np.argmax(A_test, axis=1)
        labels_test = np.argmax(test_Y, axis=1)
        accuractest_Y = np.mean(predictions_test == labels_test)

        print(f"Epoch {epoch + 1}/{epochs} - Training Accuracy: {accuractrain_Y:.4f} - Test Accuracy: {accuractest_Y:.4f} - Average Loss: {average_loss:.4f} - Learning Rate: {initial_learning_rate:.3f}")
    
    end_time = time.time()  
    total_time = end_time - start_time
    print(f"Total training time: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")

    return parameters


layer_dims = [784, 384, 10]  
parameters = train(train_X, train_Y, test_X, test_Y, layer_dims, epochs=80, batch_size=80, initial_learning_rate=0.05, dropout_rate = 0.55, lambda_l1=0.001)
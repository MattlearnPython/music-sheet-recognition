#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  4 15:25:23 2018

@author: jinzhao
"""
import numpy as np
import matplotlib.pyplot as plt  
from DL_SMIRKs_header import *
from DL_SMIRKs_read_data import *


def two_layer_model(X, y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost = False):
     
    np.random.seed(1)
    # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- 
    # Set up
    # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- 
    grads = {}
    costs = []                              
    m = X.shape[1]                           
    (n_x, n_h, n_y) = layers_dims
    
    # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- 
    # Initialize parameters dictionary,
    # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----
    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    
    # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- 
    # Loop
    # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----
    for i in range(0, num_iterations):
        # 1. Forward propagation
        A1, cache1 = linear_activation_forward(X, W1, b1, 'relu')
        A2, cache2 = linear_activation_forward(A1, W2, b2, 'sigmoid')
        
        # 2. Compute Cost
        cost = compute_cost(A2, y)
        
        # 3. Backward propagation
        dA2 = - (np.divide(y, A2) - np.divide(1 - y, 1 - A2))
        
        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, 'sigmoid')
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, 'relu')
        
        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2
        
        # 4. Update parameters
        parameters = update_parameters(parameters, grads, learning_rate)
               
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]
        
        # Prompt
        if print_cost and i % 100 == 0:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
            costs.append(cost)
            
    # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- 
    # plot the cost
    # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()


def two_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
    """
    Implements a two-layer neural network: LINEAR->RELU->LINEAR->SIGMOID.
    
    Arguments:
    X -- input data, of shape (n_x, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- dimensions of the layers (n_x, n_h, n_y)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- If set to True, this will print the cost every 100 iterations 
    
    Returns:
    parameters -- a dictionary containing W1, W2, b1, and b2
    """
    
    np.random.seed(1)
    # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- 
    # Set up
    # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- 
    grads = {}
    costs = []                              # to keep track of the cost
    m = X.shape[1]                           # number of examples
    (n_x, n_h, n_y) = layers_dims
    
    # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- 
    # Initialize parameters dictionary,
    # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----
    parameters = initialize_parameters(n_x, n_h, n_y)

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- 
    # Loop
    # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----
    for i in range(0, num_iterations):

        # 1. Forward propagation
        A1, cache1 = linear_activation_forward(X, W1, b1, 'relu')
        A2, cache2 = linear_activation_forward(A1, W2, b2, 'sigmoid')
        ### END CODE HERE ###
        
        # 2. Compute cost
        cost = compute_cost(A2, Y)

        # 3. Backward propagation
        dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))
        
        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, 'sigmoid')
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, 'relu')

        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2
        
        # 4. Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)

        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]
        
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if print_cost and i % 100 == 0:
            costs.append(cost)
       
    # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- 
    # plot the cost
    # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters
    
if __name__=="__main__":
    #===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== 
    # Set up
    #===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
    
    # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----  
    # 1. Read training set
    # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- 
    
    #folder_name = ['quarter_note', 'eighth_note', 'half_note']
    folder_name = ['quarter_note', 'eighth_note']
    label_raw_data = {"quarter_note": 1, "eighth_note":0}
    raw_data = read_data(folder_name)
    train_X, train_y = generate_data_set(raw_data, label_raw_data)
    
    # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----  
    # 2. Read testing set
    # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- 
    folder_name = ['test quarter_note', 'test eighth_note']
    label_raw_data = {"test quarter_note": 1, "test eighth_note":0}
    raw_data = read_data(folder_name)
    test_X, test_y = generate_data_set(raw_data, label_raw_data)
    
   
    # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----  
    # 3. Some parameters initialization
    # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- 
    n_x = 50 * 80     
    n_h = 7
    n_y = 1
    layers_dims = (n_x, n_h, n_y)
    
    #===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== 
    # Two-layer deep learning
    #===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
    parameters = two_layer_model(train_X, train_y, layers_dims = (n_x, n_h, n_y), num_iterations = 3000, print_cost = True)
    
    #===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== 
    # Prediction
    #===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
    pred = predict(test_X, test_y, parameters)
  
    
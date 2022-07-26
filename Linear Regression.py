# The true function
def f_true(x):
    y = 6.0 * (np.sin(x + 2) + np.sin(2*x + 4))
    return y

import numpy as np # For all our math needs
n = 750 # Number of data points
X = np.random.uniform(-7.5, 7.5, n) # Training examples, in one dimension 
B = X
e = np.random.normal(0.0, 5.0, n) # Random Gaussian noise
y = f_true(X) + e # True labels with noise

import matplotlib.pyplot as plt 
plt.figure()
# Plot the data
plt.scatter(X, y, 12, marker='o')
# Plot the true function, which is really "unknown"
x_true = np.arange(-7.5, 7.5, 0.05)
y_true = f_true(x_true)
plt.plot(x_true, y_true, marker='None', color='r')

from sklearn.model_selection import train_test_split
tst_frac = 0.3 # Fraction of examples to sample for the test set
val_frac = 0.1 # Fraction of examples to sample for the validation set

# partition (X, y) into training and test sets
X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=tst_frac, random_state=42)

# further partition (X_trn, y_trn) into training and validation sets
X_trn, X_val, y_trn, y_val = train_test_split(X_trn, y_trn, test_size=val_frac, random_state=42)

# Plot the three subsets
plt.figure()
plt.scatter(X_trn, y_trn, 12, marker='o', color='orange')
plt.scatter(X_val, y_val, 12, marker='o', color='green')
plt.scatter(X_tst, y_tst, 12, marker='o', color='blue')

###
# X float(n, ): univariate data
# d int: degree of polynomial
def polynomial_transform(X, d):
    Phi = []
    for value in X:
        z = []
        for i in range(0, d+1):
            z.append(np.power(value, i))   
        Phi.append(z)   
    Phi = np.asarray(Phi)
    return Phi

###
# Phi float(n, d): transformed data
# y float(n, ): labels
def train_model(Phi, y):
    w = (np.linalg.inv(Phi.T@Phi))@Phi.T@y
    return w

###
# Phi float(n, d): transformed data
# y float(n, ): labels
# w float(d, ): linear regression model
def evaluate_model(Phi, y, w):
    y_pred = Phi@w
    err = (y_pred - y) ** 2
    sum = 0
    for value in err:
        sum = sum + value
    mean_sq_err = sum/n
    return mean_sq_err

###1(d)
w = {} # Dictionary to store all the trained models
validationErr = {} # Validation error of the models
testErr = {} # Test error of all the models

for d in range(3, 25, 3): # Iterate over polynomial degree
    Phi_trn = polynomial_transform(X_trn, d) # Transform training data into d dimensions
    w[d] = train_model(Phi_trn, y_trn) # Learn model on training data

    Phi_val = polynomial_transform(X_val, d) # Transform validation data into d dimensions
    validationErr[d] = evaluate_model(Phi_val, y_val, w[d]) # Evaluate model on validation data
    
    Phi_tst = polynomial_transform(X_tst, d) # Transform test data into d dimensions
    testErr[d] = evaluate_model(Phi_tst, y_tst, w[d]) # Evaluate model on test data
    
# Plot all the models
plt.figure()
plt.plot(validationErr.keys(), validationErr.values(), marker='o', linewidth=3, markersize=12)
plt.plot(testErr.keys(), testErr.values(), marker='s', linewidth=3, markersize=12)
plt.xlabel('Polynomial degree', fontsize=16)
plt.ylabel('Validation/Test error', fontsize=16)
plt.xticks(list(validationErr.keys()), fontsize=12)
plt.legend(['Validation Error', 'Test Error'], fontsize=16)
plt.axis([2, 25, 0, 20])

plt.figure()
plt.plot(x_true, y_true, marker='None', linewidth=5, color='k')

for d in range(9, 25, 3):
    X_d = polynomial_transform(x_true, d)
    y_d = X_d @ w[d]
    plt.plot(x_true, y_d, marker='None', linewidth=2)
    
plt.legend(['true'] + list(range(9, 25, 3)))
plt.axis([-8, 8, -15, 15])

# X float(n, ): univariate data
# B float(n, ): basis functions
# gamma float : standard deviation / scaling of radial basis kernel

def radial_basis_transform(X, B, gamma=0.1):
    Phi = []
    for xj in X:
        z = []
        for xi in B:
            z.append(np.exp((-gamma)*(xj-xi)**2))  
        Phi.append(z)   
    Phi = np.asarray(Phi)
    return Phi

###
# Phi float(n, d): transformed data
# y float(n, ): labels
# lam float : regularization parameter
def train_ridge_model(Phi, y, lam):
    w = np.linalg.inv(Phi.T@Phi+lam*np.identity(len(Phi)))@Phi.T@y
    return w

###
# Phi float(n, d): transformed data
# y float(n, ): labels
# w float(d, ): linear regression model
def evaluate_model(Phi, y, w):
    y_pred = Phi@w
    err = (y_pred - y) ** 2
    sum = 0
    for value in err:
        sum = sum + value
    mean_sq_err = sum/n
    return mean_sq_err

w = {} # Dictionary to store all the trained models
validationErr = {} # Validation error of the models
testErr = {} # Test error of all the models

exponent = [10**m for m in range(-3, 4)]
for lam in exponent: # Iterate over polynomial degree
    Phi_trn = radial_basis_transform(X_trn, X_trn) # Transform training data into d dimensions
    w[lam] = train_ridge_model(Phi_trn, y_trn, lam) # Learn model on training data

    Phi_val = radial_basis_transform(X_val, X_trn) # Transform validation data into d dimensions
    validationErr[lam] = evaluate_model(Phi_val, y_val, w[lam]) # Evaluate model on validation data
    
    Phi_tst = radial_basis_transform(X_tst, X_trn) # Transform test data into d dimensions
    testErr[lam] = evaluate_model(Phi_tst, y_tst, w[lam]) # Evaluate model on test data

# Plot all the models
plt.figure()
plt.plot(np.log10(exponent), validationErr.values(), marker='o', linewidth=3, markersize=12)
plt.plot(np.log10(exponent), testErr.values(), marker='s', linewidth=3, markersize=12)
plt.xlabel('log(lambda)', fontsize=16)
plt.ylabel('Validation/Test error', fontsize=16)
plt.xticks(np.log10(exponent), fontsize=12)
plt.legend(['Validation Error', 'Test Error'], fontsize=16)
plt.axis([-3.5, 3.5, 0, 18])

###
plt.figure()
plt.plot(x_true, y_true, marker='None', linewidth=5, color='k')

for lam in exponent:
    X_lam = radial_basis_transform(x_true, x_true)
    w[lam] = train_ridge_model(X_lam, y_true, lam)
    y_lam = X_lam@w[lam]
    plt.plot(x_true, y_lam, marker='None', linewidth=2)

plt.legend(['true'] + exponent)
plt.axis([-8, 8, -15, 15])

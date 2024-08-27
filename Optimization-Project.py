#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import all the necessary libraries
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score


# In[2]:


# Define the path for the data
data_path = 'gisette_train.data'
labels_path = 'gisette_train.labels'

# Load the data
X_train = np.loadtxt(data_path)
y_train = np.loadtxt(labels_path)

#Inspect the data
print('Inspecting the data dimensions: \n')
print('X train dataset dimensions: ', X_train.shape)
print('y train dataset dimensions: ', y_train.shape,'\n\n')


# In[3]:


# Initialize random seed to achieve reproducability
np.random.seed(1234)

# Define the SMO class
class SMO:
    def __init__(self, C, tol, max_passes):
        self.C = C
        self.tol = tol
        self.max_passes = max_passes

    def fit(self, X, y):
        m, n = X.shape               # Get the number of training examples (m) and the number of features (n)
        self.alpha = np.zeros(m)     # Initialize Lagrange multipliers (alpha) to zero
        self.b = 0                   # Initialize the bias term (b) to zero
        self.w = np.zeros(n)         # Initialize the weight vector (w) to zero
        passes = 0                   # Number of passes without any alpha updates

        # Main training loop
        while passes < self.max_passes:
            num_changed_alphas = 0          # Track the number of alpha changes in this pass
            for i in range(m):
                E_i = self._error(X, y, i)  # Calculate the error for the i-th training example
                # Check if the i-th alpha violates the KKT conditions
                if (y[i] * E_i < -self.tol and self.alpha[i] < self.C) or (y[i] * E_i > self.tol and self.alpha[i] > 0):
                    j = self._select_j(i, m)    # Select a random j different from i
                    E_j = self._error(X, y, j)  # Calculate the error for the j-th training example
                    alpha_i_old, alpha_j_old = self.alpha[i], self.alpha[j] # Store the old values of alpha_i and alpha_j
                    if y[i] != y[j]:            # Compute L and H
                        L = max(0, self.alpha[j] - self.alpha[i])
                        H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
                    else:
                        L = max(0, self.alpha[i] + self.alpha[j] - self.C)
                        H = min(self.C, self.alpha[i] + self.alpha[j])
                    if L == H:
                        continue
                    eta = 2.0 * np.dot(X[i], X[j]) - np.dot(X[i], X[i]) - np.dot(X[j], X[j]) # Compute eta
                    if eta >= 0:
                        continue
                    self.alpha[j] -= y[j] * (E_i - E_j) / eta      # Update alpha_j
                    self.alpha[j] = np.clip(self.alpha[j], L, H)   # Clip alpha_j
                    if abs(self.alpha[j] - alpha_j_old) < 1e-5:
                        continue
                    self.alpha[i] += y[i] * y[j] * (alpha_j_old - self.alpha[j]) # Update alpha_i
                    # Compute b1 and b2
                    b1 = self.b - E_i - y[i] * (self.alpha[i] - alpha_i_old) * np.dot(X[i], X[i]) - y[j] * (self.alpha[j] - alpha_j_old) * np.dot(X[i], X[j])
                    b2 = self.b - E_j - y[i] * (self.alpha[i] - alpha_i_old) * np.dot(X[i], X[j]) - y[j] * (self.alpha[j] - alpha_j_old) * np.dot(X[j], X[j])
                    # Update b
                    if 0 < self.alpha[i] < self.C:
                        self.b = b1
                    elif 0 < self.alpha[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2
                    num_changed_alphas += 1 # Increment the number of changed alphas
            if num_changed_alphas == 0:     # Check if any alphas were changed
                passes += 1
            else:
                passes = 0
        self.w = self._compute_w(X, y)      # Compute the weight vector w

    # Predict the labels for the input data X
    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.b)

    # Calculate the error for the i-th training example
    def _error(self, X, y, i):
        return np.dot(self.w, X[i]) + self.b - y[i]

    # Select a random index j different from i
    def _select_j(self, i, m):
        j = i
        while j == i:
            j = np.random.randint(0, m)
        return j

    # Compute the weight vector w
    def _compute_w(self, X, y):
        return np.dot(X.T, self.alpha * y)

# Define hardwired parameters
C = 1.0
tol = 0.001
max_passes = 5

# Create an instance of the SMO class
smo = SMO(C, tol, max_passes)

# Fit the model to the training data
smo.fit(X_train, y_train)

print('Displaying results for hardwired values of hyperparameters C=1, tolerance=0.001 and max passes=5 (Linear Kernel)\n')
print('Estimated Weights Values Array: \n', smo.w, '\n')
print('Estimated Bias Value: \n', smo.b, '\n')


# In[4]:


print('Starting Grid Search and tuning of hyperparameters C, tolerance and max passes with Linear Kernel...\n')

# Define the parameter grid
C_values = [0.1, 1.0, 10.0]  # Different values of the regularization parameter C
tol_values = [0.001, 0.01]   # Different values for the tolerance
max_passes_values = [5, 10]  # Different values for the maximum number of passes

kf = KFold(n_splits=5)       # Prepare cross-validation using K-Folds

# Variables to store the best parameters and best score
best_score = 0
best_params = {'C': None, 'tol': None, 'max_passes': None}

# Perform grid search
for C in C_values:
    for tol in tol_values:
        for max_passes in max_passes_values:
            accuracies = []                                                           # List to store accuracy for each fold
            for train_index, val_index in kf.split(X_train):                          # Split the data into training and validation sets for the current fold
                X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
                y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

                smo = SMO(C, tol, max_passes)                          # Create an SMO instance with the current parameters
                smo.fit(X_train_fold, y_train_fold)                    # Train the SMO model on the training fold
                predictions = smo.predict(X_val_fold)                  # Predict on the validation fold
                accuracy = accuracy_score(y_val_fold, predictions)     # Calculate the accuracy on the validation fold
                accuracies.append(accuracy)                            # Store the accuracy

            avg_accuracy = np.mean(accuracies)                                                    # Calculate the average accuracy across all folds
            print(f'C: {C}, tol: {tol}, max_passes: {max_passes}, Accuracy: {avg_accuracy:.4f}')  # Print the current parameter combination and its accuracy
            if avg_accuracy > best_score:                                                         # Update the best parameters if the current average accuracy is better
                best_score = avg_accuracy 
                best_params['C'] = C
                best_params['tol'] = tol
                best_params['max_passes'] = max_passes

# Print the best parameters and the corresponding accuracy
print(f'\nBest parameters: {best_params}, Best cross-validation accuracy: {best_score:.4f}')

# Train the final model with the best parameters
smo_optimized = SMO(best_params['C'], best_params['tol'], best_params['max_passes'])
smo_optimized.fit(X_train, y_train)

print('\nEstimated Weights Values Array: \n', smo_optimized.w, '\n')
print('Estimated Bias Value: \n', smo_optimized.b, '\n\n')


# In[5]:


# Define polynomial kernel function
def polynomial_kernel(x1, x2, degree=3, coef0=1):
    return (np.dot(x1, x2) + coef0) ** degree

# SMO class with precomputed polynomial kernel matrix
class SMO:
    def __init__(self, C, tol, max_passes, kernel=polynomial_kernel, **kernel_params):
        self.C = C
        self.tol = tol
        self.max_passes = max_passes
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.X = None
        self.y = None
        self.alpha = None
        self.b = 0
        self.errors = None
        self.K = None
        self.w = None


    def fit(self, X, y):
        self.X = X
        self.y = y
        m, n = X.shape
        self.alpha = np.zeros(m)
        self.errors = np.zeros(m)
        self.b = 0
        passes = 0
        
        # Precompute the kernel matrix
        self.K = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                self.K[i, j] = self.kernel(X[i], X[j], **self.kernel_params)

        while passes < self.max_passes:
            num_changed_alphas = 0
            for i in range(m):
                E_i = self._error(i)
                if (y[i] * E_i < -self.tol and self.alpha[i] < self.C) or (y[i] * E_i > self.tol and self.alpha[i] > 0):
                    j = self._select_j(i, m)
                    E_j = self._error(j)
                    alpha_i_old, alpha_j_old = self.alpha[i], self.alpha[j]
                    if y[i] != y[j]:
                        L = max(0, self.alpha[j] - self.alpha[i])
                        H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
                    else:
                        L = max(0, self.alpha[i] + self.alpha[j] - self.C)
                        H = min(self.C, self.alpha[i] + self.alpha[j])
                    if L == H:
                        continue
                    eta = 2.0 * self.K[i, j] - self.K[i, i] - self.K[j, j]
                    if eta >= 0:
                        continue
                    self.alpha[j] -= y[j] * (E_i - E_j) / eta
                    self.alpha[j] = np.clip(self.alpha[j], L, H)
                    if abs(self.alpha[j] - alpha_j_old) < 1e-5:
                        continue
                    self.alpha[i] += y[i] * y[j] * (alpha_j_old - self.alpha[j])
                    b1 = self.b - E_i - y[i] * (self.alpha[i] - alpha_i_old) * self.K[i, i] - y[j] * (self.alpha[j] - alpha_j_old) * self.K[i, j]
                    b2 = self.b - E_j - y[i] * (self.alpha[i] - alpha_i_old) * self.K[i, j] - y[j] * (self.alpha[j] - alpha_j_old) * self.K[j, j]
                    if 0 < self.alpha[i] < self.C:
                        self.b = b1
                    elif 0 < self.alpha[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2
                    num_changed_alphas += 1
            if num_changed_alphas == 0:
                passes += 1
            else:
                passes = 0
        self.w = self._compute_w(X, y)


    def predict(self, X):
        y_predict = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            s = 0
            for alpha, y, x in zip(self.alpha, self.y, self.X):
                s += alpha * y * self.kernel(X[i], x, **self.kernel_params)
            y_predict[i] = s
        return np.sign(y_predict + self.b)

    def _compute_w(self, X, y):
        # Compute the weight vector
        return np.dot((self.alpha * y), X)

    def _error(self, i):
        return np.dot((self.alpha * self.y), self.K[:, i]) + self.b - self.y[i]

    def _select_j(self, i, m):
        j = i
        while j == i:
            j = np.random.randint(0, m)
        return j


# In[6]:


# Initialize random seed to achieve reproducability
np.random.seed(1234)

# Define the parameter grid
C_values = [1.0, 10.0]
tol_values = [0.01]
max_passes_values = [5, 10]
degree_values = [3, 4]
coef0_values = [0.1]

# Prepare cross-validation using K-Folds
kf = KFold(n_splits=3)

# Variables to store the best parameters and best score
best_score = 0
best_params = {'C': None, 'tol': None, 'max_passes': None, 'degree': None, 'coef0': None}

print('Starting Grid Search and tuning of hyperparameters C, tolerance, max passes, degree and coef0 with Polynomial Kernel...\n')

# Perform grid search
for C in C_values:
    for tol in tol_values:
        for max_passes in max_passes_values:
            for degree in degree_values:
                for coef0 in coef0_values:
                    accuracies = []
                    for train_index, val_index in kf.split(X_train):
                        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
                        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

                        smo = SMO(C=C, tol=tol, max_passes=max_passes, kernel=polynomial_kernel, degree=degree, coef0=coef0)
                        smo.fit(X_train_fold, y_train_fold)
                        predictions = smo.predict(X_val_fold)
                        accuracy = accuracy_score(y_val_fold, predictions)
                        accuracies.append(accuracy)

                    avg_accuracy = np.mean(accuracies)
                    print(f'C: {C}, tol: {tol}, max_passes: {max_passes}, degree: {degree}, coef0: {coef0}, Accuracy: {avg_accuracy:.4f}')
                    if avg_accuracy > best_score:
                        best_score = avg_accuracy
                        best_params['C'] = C
                        best_params['tol'] = tol
                        best_params['max_passes'] = max_passes
                        best_params['degree'] = degree
                        best_params['coef0'] = coef0

print(f'Best parameters: {best_params}, Best cross-validation accuracy: {best_score:.4f}')

# Train the final model with the best parameters
smo_optimized = SMO(C=best_params['C'], tol=best_params['tol'], max_passes=best_params['max_passes'], 
                    kernel=polynomial_kernel, degree=best_params['degree'], coef0=best_params['coef0'])
smo_optimized.fit(X_train, y_train)

print('\nEstimated Weights Values Array: \n', smo_optimized.w, '\n')
print('Estimated Bias Value: \n', smo_optimized.b, '\n\n')


# In[8]:


# Define the file path to store the weights and bias term
output_file = 'smo_model_parameters.txt'

# Save the weights and bias term to the text file
with open(output_file, 'w') as f:
    f.write('Weights (w):\n')
    np.savetxt(f, smo_optimized.w, delimiter=',')
    f.write('\nBias term (b):\n')
    f.write(f'{smo_optimized.b}\n')

print(f'Weights and bias terms saved to {output_file}')


# In[ ]:





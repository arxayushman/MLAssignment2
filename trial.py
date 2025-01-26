import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Clear any existing plots
plt.close('all')

# Load the dataset with explicit column names
X = pd.read_csv('logisticX (1).csv', header=None, names=['X1', 'X2'])
y = pd.read_csv('logisticY (1).csv', header=None, names=['target'])

# Standardize the features while preserving column names
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)


# Custom Logistic Regression Function
def custom_logistic_regression(X, y, learning_rate=0.1, max_iterations=1000):
    X_array = X.values
    y_array = y.values.flatten()

    # Add intercept term
    X_with_intercept = np.column_stack((np.ones(X_array.shape[0]), X_array))

    # Initialize parameters
    m, n = X_with_intercept.shape
    theta = np.zeros(n)

    # Cost function tracking
    cost_history = []

    for iteration in range(max_iterations):
        # Compute predictions using sigmoid function
        z = np.dot(X_with_intercept, theta)
        h = 1 / (1 + np.exp(-z))

        # Compute gradient
        gradient = np.dot(X_with_intercept.T, (h - y_array)) / m

        # Update parameters
        theta -= learning_rate * gradient

        # Compute log loss
        epsilon = 1e-15
        log_loss = -np.mean(y_array * np.log(h + epsilon) + (1 - y_array) * np.log(1 - h + epsilon))
        cost_history.append(log_loss)

    return theta, cost_history


# Run custom logistic regression
theta, cost_history = custom_logistic_regression(X_scaled, y)

# Print Question 1 Results
print("Question 1 Results:")
print("Final Coefficients:", theta)
print("Final Cost:", cost_history[-1])

# 1. Cost Function vs Iterations Plot
plt.figure(figsize=(10, 6))
plt.plot(range(50), cost_history[:50])
plt.title('Cost Function vs Iterations')
plt.xlabel('Iterations')
plt.ylabel('Log Loss')
plt.show()

# 2. Original Dataset with Decision Boundary
plt.figure(figsize=(10, 8))
# Plot data points
scatter = plt.scatter(X['X1'], X['X2'], c=y['target'], cmap='viridis')
plt.colorbar(scatter)

# Compute decision boundary
x_min, x_max = X['X1'].min() - 1, X['X1'].max() + 1
y_min, y_max = X['X2'].min() - 1, X['X2'].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))

# Predict for mesh grid
X_mesh = np.c_[xx.ravel(), yy.ravel()]
X_mesh_scaled = scaler.transform(X_mesh)
X_mesh_with_intercept = np.column_stack((np.ones(X_mesh.shape[0]), X_mesh_scaled))

Z = np.dot(X_mesh_with_intercept, theta)
Z = 1 / (1 + np.exp(-Z))
Z = Z.reshape(xx.shape)

plt.contour(xx, yy, Z, levels=[0.5], colors='red', linestyles='--')
plt.title('Dataset with Decision Boundary')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()

# 3. Squared Variables Dataset
plt.figure(figsize=(10, 8))
X_squared = X.copy()
X_squared['X3'] = X['X1'] ** 2
X_squared['X4'] = X['X2'] ** 2

# Standardize new dataset
X_squared_scaled = pd.DataFrame(StandardScaler().fit_transform(X_squared),
                                columns=X_squared.columns)

# Train logistic regression
lr_squared = LogisticRegression()
lr_squared.fit(X_squared_scaled, y['target'])

# Plot squared variables dataset
scatter = plt.scatter(X_squared['X3'], X_squared['X4'], c=y['target'], cmap='viridis')
plt.colorbar(scatter)
plt.title('Squared Variables Dataset')
plt.xlabel('X1²')
plt.ylabel('X2²')
plt.show()

# 4. Confusion Matrix Visualization
plt.figure(figsize=(8, 6))
# Use scikit-learn's logistic regression for consistent predictions
lr = LogisticRegression()
lr.fit(X_scaled, y['target'])
y_pred = lr.predict(X_scaled)

# Compute confusion matrix and metrics
conf_matrix = confusion_matrix(y['target'], y_pred)
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Add text annotations to the confusion matrix
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(j, i, str(conf_matrix[i, j]),
                 horizontalalignment="center",
                 color="white" if conf_matrix[i, j] > conf_matrix.max() / 2 else "black")
plt.show()

# Compute and print performance metrics
accuracy = accuracy_score(y['target'], y_pred)
precision = precision_score(y['target'], y_pred)
recall = recall_score(y['target'], y_pred)
f1 = f1_score(y['target'], y_pred)

print("\nQuestion 5 Results:")
print("Confusion Matrix:\n", conf_matrix)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)

# Note: This last print statement helps understand the coefficients for the squared variables model
print("\nSquared Variables Model Coefficients:")
print("Coefficients:", lr_squared.coef_)
print("Intercept:", lr_squared.intercept_)
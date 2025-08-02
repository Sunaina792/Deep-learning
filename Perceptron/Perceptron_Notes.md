 # Perceptron – Notes
1.What is a Perceptron?
-It's the simplest unit of a neural network that acts as a linear classifier.
-Computes a weighted sum of inputs + bias, then applies an activation function (usually step function).
-Used for binary classification problems.

 What’s Happening in the Code?
 1. Importing Libraries
```
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
```
Loads standard libraries for numerical ops, data handling, and plotting.

 2. Loading Dataset
```
df = pd.read_csv("employee_shortlisting.csv")
df.head()
```
Loads a dataset about employee shortlisting.

3. Exploratory Data Analysis (EDA)
-Visualizations with Seaborn and checking value distributions.
-Might include heatmaps, value counts, and class balance.

4. Preprocessing
-Convert categorical to numerical values using LabelEncoder
-Features (X) and target (y) are separated for training.

5. Perceptron Model from sklearn
```
from sklearn.linear_model import Perceptron
model = Perceptron()
model.fit(X_train, y_train)
```
Uses scikit-learn’s built-in Perceptron class (no manual implementation).
Trains the model using .fit()

 6. Prediction & Evaluation
```
y_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score, confusion_matrix
accuracy = accuracy_score(y_test, y_pred)
```
Makes predictions on test data
Evaluates using accuracy and confusion matrix

# Perception trics
-Generates a 2D binary classification dataset.
-class_sep=10 → makes the classes clearly separable.
-X contains feature points, and y contains labels (0 or 1).

1. Step Function
```
def step(z):
    return 1 if z > 0 else 0
```
A simple activation function.
Returns 1 if the weighted sum is positive, else returns 0.

2. Perception training function
X.shape[1] = 3 → bias + 2 features.
intercept_ (bias)
coef_ (feature weights)

3.  Plotting the Decision Boundary
```
m = -(coef_[0] / coef_[1])
b = -(intercept_ / coef_[1])
X_input = np.linspace(-3, 3, 100)
y_input = m * X_input + b
```
4. Visualization
```
plt.plot(X_input, y_input, color='red', linewidth=3)  # decision boundary
plt.scatter(X[:,0], X[:,1], c=y, cmap='winter', s=100)  # data points
```
Plots:
The red line → learned decision boundary
Points in blue and green → classified data





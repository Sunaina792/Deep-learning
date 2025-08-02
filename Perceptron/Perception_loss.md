### Perceptron Loss Function & Linear Classification: Notes
This notebook demonstrates the perceptron algorithm for linear binary classification on synthetically generated 2D data. It goes through dataset creation, algorithm implementation, model training, and plotting the learned decision boundary.

1. Data Generation
The dataset is simulated with features designed to be linearly separable (i.e., can be separated by a straight line).
```
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=100, n_features=2, n_redundant=0, n_informative=1,
    n_classes=2, n_clusters_per_class=1, random_state=41,
    hypercube=False, class_sep=15
)
```
X: Array of shape (100, 2), the two input features for each point
y: Array of shape (100,), binary class labels (0 or 1)
Purpose: Easy to visualize and to allow the perceptron to converge cleanly.

2. Data Visualization
Visual inspection of the data confirms its linear separability.
```
plt.figure(figsize=(12,4))
plt.scatter(X[:,0], X[:,1], c=y, cmap='winter', s=100)
```
Scatter plot: Each dot represents a data point, colored by class.

3. Perceptron Algorithm Implementation
A classic perceptron training loop updates the weights and bias whenever a point is misclassified.
Note: In this notebook, the label convention is 0/1, which works here due to cluster separation, but classically the labels should be -1 and +1.
```
def perceptron(X, y):
    w1 = w2 = b = 1
    lr = 0.1
    for j in range(1000):        # epochs
        for i in range(X.shape[0]): # loop over samples
            z = w1*X[i][0] + w2*X[i][1] + b
            if z*y[i] < 0:
                w1 = w1 + lr*y[i]*X[i][0]
                w2 = w2 + lr*y[i]*X[i][1]
                b  = b + lr*y[i]
    return w1, w2, b
```
w1, w2: Weights, initialized to 1.
b: Bias term, initialized to 1.
lr: Learning rate.
Update: For each point, if it's misclassified (z*y[i] < 0), the weights/bias are adjusted.
Important: To follow the standard Perceptron Algorithm exactly, convert the label array:

```
# y = np.where(y == 0, -1, 1)
```

4. Training and Parameters
The model is trained for 1000 epochs. After training, the weights/bias can be inspected:
```
w1, w2, b = perceptron(X, y)
```
These values determine the model's decision boundary.

5. Decision Boundary Calculation
The learned decision boundary is computed by solving for points where the perceptron "output" (z) equals zero:

# w1*x + w2*y + b = 0  ==>  y = -w1/w2 * x - b/w2
```
m (slope): -w1 / w2
c (intercept): -b / w2
```
Example in the file:
m: -4.53
c: -5.85
These values will differ on each run depending on the model's learned weights.

6. Plotting the Decision Boundary
The decision boundary is plotted as a red line on top of the scatter plot of data points.

```
X_input = np.linspace(-4, 3, 100)
y_input = m * X_input + c

plt.figure(figsize=(7,5))
plt.plot(X_input, y_input, color='red', linewidth=3)
plt.scatter(X[:,0], X[:,1], c=y, cmap='winter', s=100)
plt.ylim(-3, 2)
plt.grid(True)
plt.show()
```
Red line: The learned linear separator.
Dots: Data points colored by class.

7. Summary
The classic perceptron is a fundamental linear classifier.
Training: Iteratively adjusts weights to correctly classify all training samples (converges if the data is linearly separable).
Visualization: Demonstrates how the perceptron finds a separating line in simple 2D space.
# For real-world practice:
Use label convention -1/+1 for hard margin perceptron.
Longer/shorter epochs or different initialization/learning rates may affect convergence speed.


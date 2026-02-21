# 2D K-Nearest Neighbor Algorithm: A Comprehensive Analysis
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)

## 1. Introduction and Objective
The objective of this project is to evaluate distance metrics on a complex non-linear boundary.

## 2. What is K-Nearest Neighbors (KNN)?
KNN is a non-parametric, lazy learning algorithm.

## 3. The Computer Science Perspective
It requires O(N x D) time per query naively.

To optimize this, structures like KD-Trees or Ball Trees are utilized.

## 4. The Curse of Dimensionality
As dimensions increase, data becomes sparse. We stick to 2D to avoid this.

## 5. Mathematical Foundation of KNN
Majority vote determines the class prediction.

## 6. Asymptotic Error Bounds
The 1-NN error rate is bounded by the Bayes error rate.

This means 1-NN is at worst twice as bad as the absolute best possible classifier.

## 7. Distance Metrics Examined
We analyzed Euclidean, Manhattan, Cosine, Jaccard, and Hamming.

### 7.1. Euclidean Distance
Standard straight-line distance, rotation-invariant.

### 7.2. Manhattan Distance
City Block distance, highly robust to outliers.

### 7.3. Cosine Distance
Measures angle between two vectors.

### 7.4. Jaccard Distance
Used for categorical data.

### 7.5. Hamming Distance
Measures minimum substitutions for strings/vectors.

## 8. Synthetic Data Generation
We defined a custom topological surface using sine and cosine.

The equation: C(x,y) = 2*sin(x/y) + 2*cos(x*y) + 0.15

Generated 1980 random samples for training.

## 9. 3D Surface Visualization
Using mpl_toolkits.mplot3d to view the boundary.

## 10. Train, Validation, and Test Strategy
Used train_test_split and created 1500 blind test points.

## 11. Hyperparameter Tuning
Iterated K from 1 to 300 to find the optimal neighbors.

### Euclidean Results
Achieved test accuracy of 72.13%.

### Manhattan Results
Yielded 72.0% on the test set.

### Cosine Results
Outperformed others with 72.33% accuracy.

## 12. The Failure of Jaccard and Hamming
Both dropped to ~55% accuracy.

Analytical Insight: Discrete metrics fail on continuous spatial coordinates.

## 13. Comprehensive Results
Cosine: 72.33% | Euclidean: 72.13% | Manhattan: 72.00% | Jaccard & Hamming: 55.46%

## 14. Prerequisites
numpy, pandas, matplotlib, scikit-learn

## 15. Execution
Clone the repo, install dependencies, and run the script.

## 16. Conclusion
Continuous spatial data requires continuous metrics.

# 2D K-Nearest Neighbor Algorithm: A Comprehensive Analysis
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)

## 1. Introduction and Objective
The objective of this project is to evaluate distance metrics on a complex non-linear boundary.

## 2. What is K-Nearest Neighbors (KNN)?
KNN is a non-parametric, lazy learning algorithm.

## 3. The Computer Science Perspective
It requires O(N x D) time per query naively.

To optimize this, structures like KD-Trees or Ball Trees are utilized.

## 4. The Curse of Dimensionality
As dimensions increase, data becomes sparse. We stick to 2D to avoid this.

## 5. Mathematical Foundation of KNN
Majority vote determines the class prediction.

## 6. Asymptotic Error Bounds
The 1-NN error rate is bounded by the Bayes error rate.

This means 1-NN is at worst twice as bad as the absolute best possible classifier.

## 7. Distance Metrics Examined
We analyzed Euclidean, Manhattan, Cosine, Jaccard, and Hamming.

### 7.1. Euclidean Distance
Standard straight-line distance, rotation-invariant.

### 7.2. Manhattan Distance
City Block distance, highly robust to outliers.

### 7.3. Cosine Distance
Measures angle between two vectors.

### 7.4. Jaccard Distance
Used for categorical data.

### 7.5. Hamming Distance
Measures minimum substitutions for strings/vectors.

## 8. Synthetic Data Generation
We defined a custom topological surface using sine and cosine.

The equation: C(x,y) = 2*sin(x/y) + 2*cos(x*y) + 0.15

Generated 1980 random samples for training.

## 9. 3D Surface Visualization
Using mpl_toolkits.mplot3d to view the boundary.

## 10. Train, Validation, and Test Strategy
Used train_test_split and created 1500 blind test points.

## 11. Hyperparameter Tuning
Iterated K from 1 to 300 to find the optimal neighbors.

### Euclidean Results
Achieved test accuracy of 72.13%.

### Manhattan Results
Yielded 72.0% on the test set.

### Cosine Results
Outperformed others with 72.33% accuracy.

## 12. The Failure of Jaccard and Hamming
Both dropped to ~55% accuracy.

Analytical Insight: Discrete metrics fail on continuous spatial coordinates.

## 13. Comprehensive Results
Cosine: 72.33% | Euclidean: 72.13% | Manhattan: 72.00% | Jaccard & Hamming: 55.46%

## 14. Prerequisites
numpy, pandas, matplotlib, scikit-learn

## 15. Execution
Clone the repo, install dependencies, and run the script.

## 16. Conclusion
Continuous spatial data requires continuous metrics.

# 0 to 100: K-Nearest Neighbors (KNN) in 2D Space
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-0.24+-orange.svg)
![NumPy](https://img.shields.io/badge/NumPy-1.20+-blue.svg)

## 1. Introduction to the Repository
This repository is not just a simple application of a machine learning library. It is a deep dive into the mathematical, geometrical, and computational foundations of the K-Nearest Neighbors (KNN) algorithm.

We will explore everything from the theoretical proofs of asymptotic error rates to the line-by-line syntax of the Python implementation used to partition a 2D topological space.

## 2. Machine Learning Paradigm
In the broad spectrum of Artificial Intelligence, ML is divided into several paradigms. KNN falls under **Supervised Learning**, specifically solving **Classification** problems.

However, unlike neural networks or decision trees, KNN is an **Instance-Based Learning** (or lazy learning) algorithm.

### Lazy vs. Eager Learning
Eager learners construct a generalized mathematical model from the training data (e.g., finding the weights in linear regression) before receiving any new queries.

KNN is "lazy" because it completely bypasses the model-building phase. The training phase is simply storing the dataset in memory $O(1)$ time complexity. All the computational heavy lifting is deferred to the inference (prediction) phase.

## 3. The Mathematical Foundation of KNN
Let us define our feature space mathematically. Let our dataset be denoted as $D = \{(x_1, y_1), (x_2, y_2), \dots, (x_n, y_n)\}$.

Here, $x_i \in \mathbb{R}^d$ represents a feature vector in a $d$-dimensional space, and $y_i \in \{c_1, c_2, \dots, c_m\}$ represents the discrete class labels. In our specific project, $d=2$ (2D space) and $m=2$ (Binary classification: True/False).

### The Core Equation
Given a novel query point $x_q$, the algorithm identifies a subset $S \subset D$ consisting of the $k$ points closest to $x_q$. The predicted class $\hat{y}$ is determined by the mode (majority vote) of the labels in $S$:

$$ \hat{y} = \underset{v \in Y}{\text{argmax}} \sum_{i=1}^{k} I(y_i = v) $$
Where $I(\cdot)$ is the indicator function, which equals 1 if the condition is true and 0 otherwise.

## 4. Theoretical Guarantees: Cover and Hart Theorem
Why do we trust such a simple algorithm? In 1967, Thomas Cover and Peter Hart published a landmark paper proving the theoretical limits of the 1-NN algorithm.

### The Bayes Error Rate ($R^*$)
The Bayes error rate is the lowest possible error rate for any classifier on a given dataset, assuming we have absolute, perfect knowledge of the true underlying probability distributions (which is impossible in reality).

### The Theorem
Cover and Hart mathematically proved that as the number of training samples approaches infinity ($n \to \infty$), the error rate of the 1-Nearest Neighbor algorithm ($R_{1NN}$) is bounded:

$$ R^* \leq R_{1NN} \leq R^*\left(2 - \frac{c}{c-1}R^*\right) $$

Where $c$ is the number of classes. For binary classification ($c=2$), the upper bound simplifies to $2R^* - (R^*)^2$.

**Proof Intuition:** As $n \to \infty$, the nearest neighbor $x_{NN}$ to our query $x_q$ becomes infinitely close ($x_{NN} \approx x_q$). Therefore, the probability that 1-NN makes an error is the probability that the true label of $x_q$ differs from the true label of $x_{NN}$, which is governed by the Bayes risk.

**Conclusion:** 1-NN is incredibly powerful. Even without building a model, given enough data, its error rate is at worst twice the theoretical minimum possible error!

## 5. Distance Metrics: The Engine of KNN
The entire algorithm relies on how we define "closeness" or "proximity." In topology and metric spaces, a distance function $d(x, y)$ must satisfy four axioms:
1. Non-negativity: $d(x,y) \geq 0$
2. Identity of indiscernibles: $d(x,y) = 0 \iff x = y$
3. Symmetry: $d(x,y) = d(y,x)$
4. Triangle Inequality: $d(x,z) \leq d(x,y) + d(y,z)$

### The Minkowski Distance (The Generalized Metric)
Most continuous spatial metrics are special cases of the Minkowski distance ($L_p$ norm):
$$ d(x, y) = \left( \sum_{i=1}^{n} |x_i - y_i|^p \right)^{\frac{1}{p}} $$

### 5.1 Euclidean Distance ($p=2$)
When $p=2$, we get the standard Euclidean distance. It measures the shortest straight-line path between two points.
$$ d_{euc}(\mathbf{x}, \mathbf{y}) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2} $$

**Application:** This is the default in scikit-learn (`metric='minkowski', p=2`) and is highly effective for isotropic spatial data, which perfectly matches our 2D coordinate plane. Our results showed excellent accuracy (**72.13%**).

### 5.2 Manhattan Distance ($p=1$)
Also known as the Taxicab geometry or $L_1$ norm. It restricts movement to axis-aligned directions.
$$ d_{man}(\mathbf{x}, \mathbf{y}) = \sum_{i=1}^{n} |x_i - y_i| $$

**Proof of Robustness:** Because differences are not squared, Manhattan distance is significantly less sensitive to extreme outliers than Euclidean. In our topological space, it performed comparably well (**72.00%**).

### 5.3 Cosine Distance
Cosine similarity measures the cosine of the angle $\theta$ between two non-zero vectors. The distance is defined as $1 - \text{similarity}$.
$$ d_{cos}(\mathbf{x}, \mathbf{y}) = 1 - \frac{\mathbf{x} \cdot \mathbf{y}}{\|\mathbf{x}\|_2 \|\mathbf{y}\|_2} $$

**Why did it win?** In our results, Cosine distance slightly outperformed Euclidean (**72.33%**). Why? Because our generative function $C(x,y) = 2\sin(x/y) + 2\cos(xy) + 0.15$ relies heavily on the ratio $x/y$ (which defines the angle from the origin). Thus, angular separation was a highly discriminative feature!

### 5.4 Jaccard and Hamming (The Discrete Metrics)
Both of these metrics yielded terrible accuracy (**~55%**, close to random guessing). We must understand why from a mathematical standpoint.

**Hamming Distance:** Measures the proportion of coordinates that differ completely.
$$ d_{ham}(\mathbf{x}, \mathbf{y}) = \frac{1}{n} \sum_{i=1}^{n} I(x_i \neq y_i) $$

**The Mathematical Failure:** Our dataset consists of continuous `float` coordinates. The probability of two random floats being exactly equal ($x_i = y_i$) is practically zero. Therefore, Hamming distance viewed almost all points as entirely dissimilar, destroying the concept of localized neighborhoods. Discrete metrics are for text and categorical data, NOT spatial coordinates.

## 6. The Curse of Dimensionality
While KNN works beautifully in our 2D space, it fundamentally breaks down in high dimensions ($d > 10$). 

Let $r$ be the radius of a hypersphere inscribed in a hypercube of side length $2r$. As $d \to \infty$, the ratio of the volume of the hypersphere to the hypercube approaches 0.

This means in high dimensions, almost all the volume of the space is concentrated in the corners. As a result, the distance between any point and its nearest neighbor becomes almost identical to the distance to its farthest neighbor. Proximity loses its mathematical meaning.

## 7. Deep Dive: Code Syntax and Logic Breakdown
Let us analyze the Python implementation provided in this repository line by line.

### 7.1 Data Generation and The True Boundary
```python
def true_label(x,y):
    c = 2*np.sin(x/y) + 2*np.cos(x*y) + 0.15
    return c > 0.5
```
This is our oracle. We define a highly non-linear, trigonometric boundary. The term $\sin(x/y)$ creates radial patterns, while $\cos(xy)$ creates hyperbolic ripples.

### 7.2 Creating the Space with `meshgrid`
```python
x = np.linspace(-10, 10, N)
X, Y = np.meshgrid(x, y)
```
`np.linspace` creates evenly spaced arrays. `np.meshgrid` is a crucial NumPy function that takes two 1D arrays and creates two 2D matrices representing every possible $(X, Y)$ coordinate combination. This is necessary for plotting surfaces.

### 7.3 Visualization using `mplot3d`
```python
wf = plt.axes(projection='3d')
wf.plot_surface(Xfroshow, Yfroshow2, Zforshow, alpha=1, color='yellow')
```
We project our complex mathematical function into a 3D space, where the Z-axis represents the output $C(x,y)$. We then intersect this surface with a flat plane at $Z = 0.5$ (colored black in the code) to visually demonstrate where the decision boundary splits the space.

### 7.4 Random Sampling and Array Population
```python
rndx = 20*np.random.rand(1980) - 10
```
`np.random.rand` generates values between $[0, 1)$. By multiplying by 20 and subtracting 10, we scale and shift the distribution to our target domain: $[-10, 10]$.

We then iterate through these points, passing them through `true_label()` to separate them into Red (`True`) and Blue (`False`) lists. This forms our raw dataset.


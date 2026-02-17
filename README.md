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


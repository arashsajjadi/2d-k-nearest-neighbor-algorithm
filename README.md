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


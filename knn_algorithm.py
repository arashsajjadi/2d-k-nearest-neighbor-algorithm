import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def true_label(x,y):
    c = 2*np.sin(x/y) + 2*np.cos(x*y) + 0.15
    return c > 0.5
    
# KNN Implementation and Metric evaluation
# Full code available in this repository.

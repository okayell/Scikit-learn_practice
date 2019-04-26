import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import neighbors
from sklearn.model_selection import cross_val_score 
import matplotlib.pyplot as plt

iris = datasets.load_iris()

X = pd.DataFrame(iris.data, columns=iris.feature_names)
X.columns = ["sepal_length","sepal_width","petal_length","petal_width"]
target = pd.DataFrame(iris.target, columns=["target"])
y = target["target"]

Ks =np.arange(1, round(0.2*len(X) + 1 ))
accuracies = []
accuracies2 = []
for k in Ks:
    knn = neighbors.KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn,X,y,scoring="accuracy",cv=12)
    
    accuracies.append(scores.mean())
    accuracies2.append(scores)    
    
plt.plot(Ks,accuracies)
plt.show()
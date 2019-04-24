import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn import cluster
import matplotlib.pyplot as plt

iris = datasets.load_iris()

X = pd.DataFrame(iris.data, columns=iris.feature_names)
X.columns = ["sepal_length","sepal_width","petal_length","petal_width"]
y = iris.target
k = 3

kmeans = cluster.KMeans(n_clusters=k, random_state=12)
kmeans.fit(X)
print("K-means Classification:\n", kmeans.labels_)

pred_y = np.choose(kmeans.labels_, [2,0,1]).astype(np.int64)
print("K-means Fix Classification:\n", pred_y)
print("Real Classification:\n", y)

import sklearn.metrics as sm
print(sm.accuracy_score(y,pred_y))
print(sm.confusion_matrix(y,pred_y))

colmap = np.array(['r','g','y'])
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.subplots_adjust(hspace = 0.5)
plt.scatter(X["petal_length"],X["petal_width"],color=colmap[y])
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.title("Real Classification")
plt.subplot(1,2,2)
plt.scatter(X["petal_length"],X["petal_width"],color=colmap[pred_y])
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.title("K-means Classification")
plt.show()
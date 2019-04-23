import pandas as pd
import numpy as np
from sklearn import neighbors


X = pd.DataFrame({"durability":[7,7,3,1],"strength":[7,4,4,4]})
y = np.array([0,0,1,1])
k = 3
knn = neighbors.KNeighborsClassifier(n_neighbors=k)
knn.fit(X,y)

new_tissue = pd.DataFrame(np.array([[3,7]]))
pred = knn.predict(new_tissue)
print(pred)
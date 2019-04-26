from sklearn import datasets

iris = datasets.load_iris()
print(iris.keys())
print(iris.data.shape)
print(iris.feature_names)
print(iris.DESCR)

import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split

X = pd.DataFrame(iris.data, columns=iris.feature_names)
target = pd.DataFrame(iris.target, columns=["target"])
y = target["target"]

XTrain, XTest, yTrain, yTest = train_test_split(X,y,test_size=0.33,random_state=1)

dtree = tree.DecisionTreeClassifier(max_depth = 8)
dtree.fit(XTrain, yTrain)
print("準確率:", dtree.score(XTest, yTest))

print(dtree.predict(XTest))
print(yTest.values)

preds = dtree.predict_proba(X=XTest)

print(pd.crosstab(preds[:,1], columns=iris.feature_names))

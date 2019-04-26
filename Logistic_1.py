import pandas as pd
import numpy as np
from sklearn import preprocessing, linear_model

titanic = pd.read_csv("./data/titanic.csv")
#print(titanic.info())

age_median = np.nanmedian(titanic["Age"])
print("年齡中位數:", age_median)
new_age = np.where(titanic["Age"].isnull(), age_median, titanic["Age"])
titanic["Age"] = new_age
label_encoder = preprocessing.LabelEncoder()
encoded_class = label_encoder.fit_transform(titanic["PClass"])

X = pd.DataFrame([encoded_class,titanic["SexCode"],titanic["Age"]]).T
y = titanic["Survived"]
print(pd.DataFrame(titanic["Age"]))
print(pd.DataFrame([encoded_class,titanic["Age"]]))
'''
logistic = linear_model.LogisticRegression()
logistic.fit(X,y)
print("迴歸係數:", logistic.coef_)
print("截距:", logistic.intercept_)

preds = logistic.predict(X)
print(pd.crosstab(preds,titanic["Survived"]))

print((804+265) / (804+185+59+265))
print(logistic.score(X,y))
'''


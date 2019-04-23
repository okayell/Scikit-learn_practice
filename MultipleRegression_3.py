import numpy as np
import pandas as pd
from sklearn import datasets    # 載入資料集
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

boston = datasets.load_boston()
#print(boston.keys())
#print(boston.data.shape)
#print(boston.feature_names)
#print(boston.DESCR)

X = pd.DataFrame(boston.data, columns=boston.feature_names)
print(X.head())
target = pd.DataFrame(boston.target, columns=["MEDV"])
print(target.head())
y = target["MEDV"]

lm = LinearRegression()
lm.fit(X,y)
print("迴歸係數:", lm.coef_)
print("截距:", lm.intercept_)
coef = pd.DataFrame(boston.feature_names, columns=["features"])
coef["estimatedCoefficients"] = lm.coef_
print(coef)

plt.scatter(X.RM, y)
plt.xlabel("Average number of rooms per dwelling(RM)")
plt.ylabel("Housing Price(MEDV)")
plt.title("Relationship between RM and Price")
plt.show()

predicted_price = lm.predict(X)
print(predicted_price[0:5])

plt.scatter(y,predicted_price)
plt.xlabel("Price")
plt.ylabel("Predicted Price")
plt.title("Price vs Predicted Price")
plt.show()

MSE = np.mean((y-predicted_price)**2)
print("MSE:",MSE)
print("R-squared:", lm.score(X,y))

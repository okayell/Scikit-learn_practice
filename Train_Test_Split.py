import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

boston = datasets.load_boston()

X = pd.DataFrame(boston.data, columns=boston.feature_names)
target = pd.DataFrame(boston.target, columns=["MEDV"])
y = target["MEDV"]

XTrain,XTest,yTrain,yTest = train_test_split(X,y,test_size=0.33,random_state=5)

lm = LinearRegression()
lm.fit(XTrain,yTrain)

pred_train = lm.predict(XTrain)
pred_test = lm.predict(XTest)
plt.scatter(yTest,pred_test)
plt.xlabel("Price")
plt.ylabel("Predicted Price")
plt.title("Price vs Prediced Price")
plt.show()


MSE_train = np.mean((yTrain-pred_train)**2)
MSE_test = np.mean((yTest-pred_test)**2)
print("訓練資料的MSE:", MSE_train)
print("測試資料的MSE:", MSE_test)
print("訓練資料的R-squared:", lm.score(XTrain, yTrain))
print("測試資料的R-squared:", lm.score(XTest, yTest))

'''
plt.scatter(pred_train, pred_train-yTrain, "b", alpha=0.5, label="Training Data")
plt.scatter(pred_test, pred_test-yTest, "r", label="Test Data")
plt.hlines(y=0, xmin=0, xmax=50)
plt.title("Residual Plot")
plt.ylabel("Residual Value")
plt.legend()
plt.show()
'''
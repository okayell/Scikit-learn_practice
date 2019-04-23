import numpy as np
import pandas as pd 
from sklearn.linear_model import LinearRegression

# ------複迴歸模型 y = a + bX(多個解釋變數)------ #

waist_heights = np.array([[67,160],[68,165],[70,167],[65,170],[80,165],
                          [85,167],[78,178],[79,182],[95,175],[89,172]])
weights = np.array([50,60,65,65,70,75,80,85,90,81])

X = pd.DataFrame(waist_heights, columns=["Waist","Height"])
target = pd.DataFrame(weights, columns=["Weight"])
y = target["Weight"]

lm = LinearRegression()
lm.fit(X,y)
print("迴歸係數:", lm.coef_)
print("截距:", lm.intercept_)

new_waist_heights = pd.DataFrame(np.array([[66,164],[82,172]]))
predicted_weights = lm.predict(new_waist_heights)
print(predicted_weights)
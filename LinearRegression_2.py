import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# -----------簡單線性迴歸模型 y = a + bX----------------- #

# 建立 氣溫 與 營業額 陣列資料
heights = np.array([147.9,163.5,159.8,155.1,163.3,
                    158.7,172.0,161.2,153.9,161.6])
weights = np.array([41.7,60.2,47.0,53.2,48.3,
                    55.2,58.3,49.0,46.7,52.5])
# 建立 X 解釋變數的DataFrame物件
X = pd.DataFrame(heights, columns=["Height"])
# 建立 target 反應變數(y)的DataFrame物件
target = pd.DataFrame(weights, columns=["Weight"])
y = target["Weight"]
# 建立 lm 線性迴歸物件
lm = LinearRegression()
# 呼叫 fit() 函數來訓練模型
lm.fit(X,y)
print("迴歸係數:", lm.coef_)
print("截距:", lm.intercept_)


# ------------------使用模型預測------------------------ #
new_heights = pd.DataFrame(np.array([150,160,170]))
predicted_weights = lm.predict(new_heights)
print(predicted_weights)


# ------------------繪製迴歸縣------------------------ #
import matplotlib.pyplot as plt

plt.scatter(heights,weights)
regression_weights = lm.predict(X)
plt.plot(heights,regression_weights, "b")
plt.plot(new_heights, predicted_weights, "ro", markersize=10)
plt.show()
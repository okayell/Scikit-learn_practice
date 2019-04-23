import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# 簡單線性迴歸模型 y = a + bX
# -----------簡單線性迴歸模型 y = a + bX----------------- #
# 建立 氣溫 與 營業額 陣列資料
temperatures = np.array([29,28,34,31,
                         25,29,32,31,
                         24,33,25,31,
                         26,30])
drink_sales = np.array([7.7,6.2,9.3,8.4,
                        5.9,6.4,8.0,7.5,
                        5.8,9.1,5.1,7.3,
                        6.5,8.4])
# 建立 X 解釋變數的DataFrame物件
X = pd.DataFrame(temperatures, columns=['Temperature'])
# 建立 target 反應變數(y)的DataFrame物件
target = pd.DataFrame(drink_sales,columns=["Drink_Sales"])
y = target["Drink_Sales"]
# 建立 lm 線性迴歸物件
lm = LinearRegression()
# 呼叫 fit() 函數來訓練模型
lm.fit(X,y)     #第一個參數: 解釋變數, 第二個參數: 反應變數
print("迴歸係數:", lm.coef_)      # 顯示 迴歸係數(b)
print("截距:", lm.intercept_)    # 顯示 截距(a)
# ------------------模型建立完成------------------------ #

# 輸入新資料溫度預測營業額
new_temperatures = pd.DataFrame(np.array([26,30]))  # 新溫度的DataFrame物件
predicted_sales = lm.predict(new_temperatures)      # 利用 predict() 預測營業額
print(predicted_sales)
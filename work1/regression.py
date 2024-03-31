import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import seaborn as sns


def PolynomialRegression(degree):
    return Pipeline(
        [
            ("poly", PolynomialFeatures(degree)),
            ("std_scaler", StandardScaler()),
            ("line_reg", LinearRegression()),
        ]
    )


# 初始化数据
scores_test = []
scores_train = []
mse_train = []
mse_test = []
degrees = []

# 数据导入
# os.chdir("homework\work")
# # print(os.getcwd())
file = "complex_nonlinear_data.csv"
df = pd.read_csv(file)
data = df.iloc[:, 0:2]
train_x = data["x"]
train_y = data["y_complex"]

file2 = "new_complex_nonlinear_data.csv"
df2 = pd.read_csv(file2)
data2 = df2.iloc[:, 0:2]
test_x = data2["x_new"]
test_y = data2["y_new_complex"]


# IQR
sns.boxenplot(train_y)
Q1 = np.percentile(train_y, 25, interpolation="midpoint")
Q3 = np.percentile(train_y, 75, interpolation="midpoint")
IQR = Q3 - Q1
upper_bound = Q3 + 1 * IQR
lower_bound = Q1 - 1 * IQR
upper_indices = np.where(train_y > upper_bound)[0]
lower_indices = np.where(train_y < lower_bound)[0]
train_x = train_x.drop(list(np.concatenate([upper_indices, lower_indices])))
train_y = train_y.drop(list(np.concatenate([upper_indices, lower_indices])))
train_x = train_x.dropna().reset_index(drop=True)
train_y = train_y.dropna().reset_index(drop=True)
X = train_x.values.reshape(-1, 1)

# 10折交叉验证
Kfold = KFold(n_splits=10, shuffle=True)
for degree in range(30):
    for train_index, test_index in Kfold.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = train_y[train_index], train_y[test_index]
        # 回归模型
        poly_reg = PolynomialRegression(degree)
        poly_reg.fit(X_train, Y_train)
        y_predict_train = poly_reg.predict(X_train)
        y_predict_test = poly_reg.predict(X_test)
        scores_train.append(mean_squared_error(Y_train, y_predict_train))
        scores_test.append(mean_squared_error(Y_test, y_predict_test))
    mse_test.append(np.mean(scores_test))
    mse_train.append(np.mean(scores_train))
# 最优次项
print(mse_test.index(min(mse_test)))
# 学习曲线
fig1 = plt.figure("figure1")
plt.plot(mse_train, "r", label="mse_train")
plt.plot(mse_test, "b", label="mse_test")
plt.title("Figure1")
plt.legend()
plt.xlabel("Degree of polynomial")
plt.ylabel("MSE")
# 训练测试
poly_reg = PolynomialRegression(mse_test.index(min(mse_test)))
poly_reg.fit(data["x"].values.reshape(-1, 1), data["y_complex"])
Y_predict = poly_reg.predict(data["x"].values.reshape(-1, 1))
print(mean_squared_error(data["y_complex"], Y_predict))

Y_new_predict = poly_reg.predict(test_x.values.reshape(-1, 1))
print(mean_squared_error(test_y, Y_new_predict))

fig2 = plt.figure("figure2")
plt.scatter(train_x, train_y, c="b")
plt.plot(np.sort(data["x"]), Y_predict[np.argsort(data["x"])], "r")
plt.title("Figure2")
plt.xlabel("x")
plt.ylabel("y_complex")

fig3 = plt.figure('figure3')
plt.scatter(test_x, test_y, c="b")
plt.plot(np.sort(data2["x_new"]), Y_new_predict[np.argsort(data2["x_new"])], "r")
plt.title("Figure3")
plt.xlabel("x_new")
plt.ylabel("y_new_complex")
plt.show()

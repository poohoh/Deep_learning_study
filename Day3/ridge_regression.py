import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

degree = 1
lambda_value = 1

# ridge regression
def ridge_reg(x, y, degree, lambda_value):
    # x값을 가지는 행렬 A와 y값을 가지는 행렬 b 생성
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    A = poly.fit_transform(x.reshape(-1, 1))
    b = y.reshape((-1, 1))

    # 최적해 계산 - 상수항부터
    w = np.linalg.inv(A.T @ A + lambda_value * np.identity(A.shape[1])) @ A.T @ b

    return w

# 방정식 값 계산
def calculate_equation(a, w):
    result = 0
    for degree, n in enumerate(w):
        result += n * a ** degree

    return result

# 주어진 데이터
x = np.array([0.8147, 0.9058, 0.127, 0.9134, 0.6324, 0.0975, 0.2785, 0.5469, 0.9575, 0.9649])
y = np.array([0.8576, 0.9706, 0.2572, 0.8854, 0.8003, 0.1419, 0.4218, 0.9157, 0.7922, 0.9595])

# polynomial regression 수행
w = ridge_reg(x, y, degree, lambda_value).squeeze()
print(w)

# 그래프 그리기
a = np.linspace(0, 1, 100)
b = calculate_equation(a, w)
plt.scatter(x, y)
plt.plot(a, b)
plt.show()

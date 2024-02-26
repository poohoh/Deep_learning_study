import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score

x = np.array([0.8147, 0.9058, 0.127, 0.9134, 0.6324, 0.0975, 0.2785, 0.5469, 0.9575, 0.9649])
y = np.array([0.8576, 0.9706, 0.2572, 0.8854, 0.8003, 0.1419, 0.4218, 0.9157, 0.7922, 0.9595])
classes = np.array([1, 1, 0, 1, 0, 0, 0, 1, 0, 1])

features = np.column_stack((x, y))

# SVM
classifier = svm.SVC(kernel='linear', C=100.0)
classifier.fit(features, classes)  # train classifier

# plotting
plt.scatter(x, y, c=classes, cmap=plt.cm.Paired, marker='o', edgecolors='k')
ax = plt.gca()  # 현재 그림의 축. 축의 범위 설정, 레이블 추가 등

# decision boundary
xlim = ax.get_xlim()  # 축의 범위
ylim = ax.get_ylim()

xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50),  # 현재 범위를 등간격으로 나눈 x,y 값을 생성하고, 2차원 그리드 생성
                     np.linspace(ylim[0], ylim[1], 50))  # xx, yy는 각각 x축과 y축의 좌표 그리드. 각 값에 대해 decision function의 값을 계산하기 위함
Z = classifier.decision_function(np.c_[xx.ravel(), yy.ravel()])  # xx와 yy를 1차원 배열로 펼친 후, 열로 결합하여 2차원 배열 생성하고, 입력 데이터에 대한 decision function 값 계산
Z = Z.reshape(xx.shape)  # decision function의 값을 2차원 배열로 reshape

ax.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

# support vectors
ax.scatter(classifier.support_vectors_[:, 0], classifier.support_vectors_[:, 1],
           s=100, facecolors='none', edgecolors='k')

plt.title('SVM Decision Boundary')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()

# accuracy, precision, recall
pred = classifier.predict(features)
accuracy = accuracy_score(classes, pred)
precision = precision_score(classes, pred)
recall = recall_score(classes, pred)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')

weights = classifier.coef_
bias = classifier.intercept_

print(f'Weights: {weights}')
print(f'Bias: {bias}')
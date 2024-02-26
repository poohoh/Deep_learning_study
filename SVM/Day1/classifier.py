import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score

x = np.array([0.8147, 0.9058, 0.127, 0.9134, 0.6324, 0.0975, 0.2785, 0.5469, 0.9575, 0.9649])
y = np.array([0.8576, 0.9706, 0.2572, 0.8854, 0.8003, 0.1419, 0.4218, 0.9157, 0.7922, 0.9595])
classes = np.array([1, 1, 0, 1, 0, 0, 0, 1, 0, 1])

features = np.column_stack((x, y))

# SVM
classifier = svm.SVC(kernel='linear', C=1000.0)
classifier.fit(features, classes)

# plotting
plt.scatter(x, y, c=classes, cmap=plt.cm.Paired, marker='o', edgecolors='k')
ax = plt.gca()

# decision boundary
xlim = ax.get_xlim()
ylim = ax.get_ylim()

xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50),
                     np.linspace(ylim[0], ylim[1], 50))
Z = classifier.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

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
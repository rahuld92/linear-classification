from sklearn.datasets import make_regression
import matplotlib.pylab as plt
import numpy as np

x, y = make_regression(n_samples=100, n_features=1, noise=22)

x_mean = np.mean(x)
y_mean = np.mean(y)

num = 0
den = 0
for i in range(len(x)):
    num += (x[i] - x_mean) * (y[i] - y_mean)
    den += (x[i] - x_mean) ** 2
coefficient = num / den
intercept = y_mean - (coefficient * x_mean)

y_pred = intercept + coefficient * x

plt.scatter(x, y)
plt.plot(x, y_pred, 'r')
plt.show()

ssr = 0

for i in range(len(x)):
    ssr += (y[i] - y_pred[i]) ** 2

print(f'Sum of Square Error: {ssr}')

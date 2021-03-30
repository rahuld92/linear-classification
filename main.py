from sklearn.datasets import make_regression
import matplotlib.pylab as plt
import numpy as np

x_data, y_data = make_regression(n_samples=100, n_features=2, noise=22)


plt.scatter(x_data, y_data)
x = np.c_[np.ones(x_data.shape[0]), x_data]
X = np.linalg.pinv(x)

w = X @ y_data
w = np.insert(w, 0, 1)
w = w.tolist()

print(w)
y_pred = (-w[0] - w[1] * x_data[:, :0])
slope = -(w[1]) / (w[0])
intercept = -w[0] / w[1]

y_pred = (slope * x) + intercept
plt.plot(x, y_pred)

plt.show()
ssr = 0
for i in range(len(y_data)):
    ssr += (y_data[i] - y_pred[i])**2

print(f'Sum of Error Squared {ssr}')

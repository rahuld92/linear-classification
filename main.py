from sklearn.datasets import make_regression
import matplotlib.pylab as plt
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns


def generate_dataset_simple(beta, n, std_dev):
    # Generate x as an array of `n` samples which can take a value between 0 and 100
    x = np.zeros(np.random() * 100)
    # Generate the random error of n samples, with a random value from a normal distribution, with a standard
    # deviation provided in the function argument
    e = np.random.randn(n) * std_dev
    # Calculate `y` according to the equation discussed
    y = x * beta + e
    return x, y


x, y = make_regression(n_samples=150, n_features=1, noise=20)
# x, y = generate_dataset_simple(10, 50, 100)

x_mean = np.mean(x)
y_mean = np.mean(y)

# Calculate Intercept
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
sst = 0

for i in range(len(x)):
    ssr += (y[i] - y_pred[i]) ** 2
    sst += (y[i] - y_mean) ** 2

print(f'Sum of Square Error: {ssr / sst}')

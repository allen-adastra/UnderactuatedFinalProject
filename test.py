import matplotlib.pyplot as plt
import numpy as np

# Test arctan function approximation.
xs = np.linspace(-0.2, 0.2, 100)
exact = np.arctan(xs)
approx = xs

f, (ax1, ax2) = plt.subplots(2, 1, sharex = True)
ax1.plot(xs, exact)
ax1.plot(xs, approx)
ax2.plot(xs, [a - e for a, e in zip(approx, exact)])
plt.show()
import matplotlib.pylab as plt
import numpy as np

x = np.linspace(0, 10*np.pi, 1000000)
plt.plot(x, np.sin(x))
plt.show()

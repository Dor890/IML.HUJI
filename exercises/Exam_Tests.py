import numpy as np

print(np.tril(np.ones(shape=10), -1) - np.triu(np.ones(shape=10)))

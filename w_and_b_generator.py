import pandas as pd
import numpy as np
np.set_printoptions(threshold=np.inf)
## Initializing random weights
W1 = np.round(np.random.rand(784, 10),4)  # each row is weight for corresponding pixel
B1 = np.round(np.random.rand(10),4)
W2 = np.round(np.random.rand(10, 10),4)  # layer 2
B2 = np.round(np.random.rand(10),4)
print(W1)
np.savetxt("C:/Users/harik/Pictures/W1.txt",W1)
np.savetxt("C:/Users/harik/Pictures/W2.txt",W2)
np.savetxt("C:/Users/harik/Pictures/B1.txt",B1)
np.savetxt("C:/Users/harik/Pictures/B2.txt",B2)


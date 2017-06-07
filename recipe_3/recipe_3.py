#############################################################
# What makes a good feature? - Machine Learning Recipes #3  #
# https://www.youtube.com/watch?v=N9fDIAflCMY               #
#############################################################

import numpy as np
import matplotlib.pyplot as plt

greyhounds = 500
labs = 500

np.random.seed(8)

# randn returns a float between (-inf, +inf) following a normal distribution with mean 0 and standard deviation 1
grey_height = 28 + 4 * np.random.randn(greyhounds)
lab_height = 24 + 4 * np.random.randn(labs)

plt.hist([grey_height, lab_height],
         stacked=True,
         color=["orange", "blue"],
         label=["Greyhounds", "Labs"])
plt.legend(prop={"size": 10})
plt.xlabel("dog height")
plt.ylabel("# of dogs")
plt.show()

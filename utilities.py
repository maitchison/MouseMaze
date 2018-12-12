"""
General utility functions
"""

import numpy as np

# utility functions
def clip(x,low,high):
    if x < low: return low
    if x > high: return high
    return x


def smooth(X, alpha=0.95):
    Y = []
    y = X[0]
    for x in X:
        y = alpha * y + (1-alpha) * x
        Y.append(y)
    return Y


def bits_to_int(bits):
    return int("".join([str(1 if i else 0) for i in bits]),2)

def int_to_bits(x, fill):
    return [int(c) for c in bin(x)[2:].zfill(fill)]

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()



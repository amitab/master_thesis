from __future__ import division
import math
import numpy as np


def signature_bit(data, planes):
    """
	LSH signature generation using random projection
	Returns the signature bits for two data points.
	The signature bits of the two points are different
 	only for the plane that divides the two points.
 	"""
    sig = 0
    for p in planes:
        sig <<= 1
        if np.dot(data, p) >= 0:
            sig |= 1
    return sig


def bitcount(n):
    """
	gets the number of bits set to 1
	"""
    count = 0
    while n:
        count += 1
        n = n & (n - 1)
    return count


def length(v):
    """returns the length of a vector"""
    return math.sqrt(np.dot(v, v))
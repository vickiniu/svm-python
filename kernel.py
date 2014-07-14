'''
Vicki Niu, July 2k14
Description: implements some kernel functions 
(see the list here: http://en.wikipedia.org/wiki/Support_vector_machine#Nonlinear_classification)
Packages: numpy & the linalg module from numpy
'''

import numpy
from numpy import linalg

class kernel(object):
    
    @staticmethod
    def linear():
        def f(x, y):
            return numpy.inner(x, y)
        return f
        
    @staticmethod
    def gaussian(sigma):
        def f(x, y):
            return numpy.exp(-linalg.norm(x - y)** 2 / (2 * (sigma ** 2)))
        return f
        
    @staticmethod
    def polynomial(dim, offset):
        def f(x, y):
            return (offset + numpy.dot(x, y)) ** dim
        return f
        
    @staticmethod
    def inhomogenous_polynomial(dim, offset):
        return kernel.polynomial(dim = dim, offset = 1.0)
        
    @staticmethod
    def homogenous_polynomial(dim, offset):
        return kernel.polynomial(dim = dim, offset = 0.0)

    @staticmethod
    def tanh(kappa, c):
        def f(x, y):
            return numpy.tanh(kappa * numpy.dot(x, y) + c)
        return f
            

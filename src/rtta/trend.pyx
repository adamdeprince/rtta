import sys
import numpy as np
cimport numpy as cnp
cimport cython
cnp.import_array()

cdef class SMAIndicator():
    """SMA - Simple Moving Average

    Args:
      window(int): n period
      fillna(bool): if True, fill nan values
    """

    cdef bint first_pass
    cdef object history
    cdef double[:] history_view
    cdef int index
    cdef int window
    cdef bint fillna
    cdef double tally
    
    def __init__(self, int window, bint fillna=False):
        self.history = np.zeros(window)
        self.history_view = self.history
        self.first_pass = True
        self.index = 0
        self.window = window
        self.fillna = fillna
        self.tally = 0
        
    @cython.boundscheck(False) # turn off bounds-checking for entire function
    @cython.wraparound(False)
    cpdef double update(self, float value):
        self.tally -= self.history_view[self.index]
        self.tally += value
        self.history_view[self.index] = value

        self.index += 1

        if self.index == self.window:
            self.index = 0
            self.first_pass = False
            
        if self.first_pass:
            if not self.fillna:
                return np.nan
            return self.tally / self.index
        return self.tally/self.window



cdef class EMAIndicator():
    """SMA - Simple Moving Average

    Args:
      window(int): n period
      fillna(bool): if True, fill nan values
    """

    cdef bint first_pass
    cdef int index
    cdef int window
    cdef bint fillna
    cdef double last_value
    cdef double weighted_multiplier
    cdef double inverted_multiplier
    
    def __init__(self, int window, bint fillna=False):
        self.first_pass = True
        self.index = 0
        self.window = window
        self.fillna = fillna
        self.last_value = 0
        self.weighted_multiplier = 2.0 / (1.0 + float(window))
        self.inverted_multiplier = 1 - self.weighted_multiplier

    cpdef double update(self, float value):
        self.last_value = self.weighted_multiplier * value + self.last_value * self.inverted_multiplier
        

        self.index += 1
        if self.index == self.window:
            self.index = 0
            self.first_pass = False

        if self.first_pass and not self.fillna:
            return np.nan

        return self.last_value

    
cdef class MACDIndicator():
    """MCAD - Moving average convergence divergence.
    https://en.wikipedia.org/wiki/MACD

    MACD(a, b, c)
    
    Args:
      a, b, c: The EMA period parameters
    """

    cdef EMAIndicator a
    cdef EMAIndicator b
    cdef EMAIndicator c

    def __init__(self, int a, int b, int c, bint fillna=False):
        self.a = EMAIndicator(window=a, fillna=False)
        self.b = EMAIndicator(window=b, fillna=False)
        self.c = EMAIndicator(window=c, fillna=False)

    def update(self, value):
        return self.c.update(self.a.update(value) - self.b.update(value))



import sys
import numpy as np
cimport numpy as cnp
cimport cython
cnp.import_array()

cdef class Summation():
    """Summation - Summation acorss a fixed window.

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
            return self.tally
        return self.tally


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

    cpdef double update(self, double value):
        if self.first_pass and self.index == 0:
            self.last_value = value
        self.last_value = self.weighted_multiplier * value + self.last_value * self.inverted_multiplier
        self.index += 1
        if self.index == self.window:
            self.index = 0
            self.first_pass = False

        if self.first_pass and not self.fillna:
            return np.nan

        return self.last_value

    
cdef class MACD():
    """MCAD - Moving average convergence divergence.
    https://en.wikipedia.org/wiki/MACD

    MACD(a, b, c)
    
    Args:
      a, b, c: The EMA period parameters
    """

    cdef EMAIndicator a
    cdef EMAIndicator b
    cdef EMAIndicator c
    cdef long counter
    cdef bint fillna
    cdef int window    

    def __init__(self, int a=12, int b=26, int c=9, bint fillna=False):
        """
        https://school.stockcharts.com/doku.php?id=technical_indicators:moving_average_convergence_divergence_macd
        """
        self.a = EMAIndicator(window=a, fillna=True)
        self.b = EMAIndicator(window=b, fillna=True)
        self.c = EMAIndicator(window=c, fillna=fillna)
        self.counter = 0
        self.fillna = fillna
        self.window = max(a,b) +c 

    cpdef double update(self, value):
        cdef double retval
        try:
            retval = self.c.update(self.a.update(value) - self.b.update(value))
            if not self.fillna and self.counter < self.window:
                return np.nan
            return retval
        finally:
            self.counter += 1 

cdef class MassIndex():
    cdef EMAIndicator single
    cdef EMAIndicator double
    cdef Summation summation
    cdef long counter
    cdef int window
    cdef bint fillna
    
    def __init__(self, int single=9, int double=9, int summation=25, bint fillna=False):
        self.single = EMAIndicator(window=single, fillna=True)
        self.double = EMAIndicator(window=double, fillna=True)
        self.summation = Summation(window=summation, fillna=True)
        self.counter = 0
        self.window = max(single, double) + summation
        self.fillna = fillna

    cpdef double update(self, high, low):
        cdef double retval
        cdef double single
        cdef double double_
        try:
            if high < low:
                high, low = low, high
                
            single = self.single.update(high-low)
            double_ = self.double.update(single)
            
            retval = self.summation.update(single/double_)
            if not self.fillna and self.counter < self.window:
                return np.nan
            return retval
        finally:
            self.counter += 1

        
        
    



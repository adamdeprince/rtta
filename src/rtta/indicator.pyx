import sys
import numpy as np
cimport numpy as cnp
cimport cython
cnp.import_array()
from collections import namedtuple



        
cdef class AwesomeOscillatorIndicator:
    cdef SMA oscillator_1
    cdef SMA oscillator_2
    cdef long counter
    cdef int window

    def __init__(self, int window_1 = 34, int window_2 = 5, bint fillna = True):
        self.oscillator_1 = SMA(window=window_1, fillna=True)
        self.oscillator_2 = SMA(window=window_2, fillna=True)
        self.counter = 0
        self.window = max(window_1, window_2) if not fillna else 0

    cpdef double update(self, double high, double low):
        self.counter += 1

        if high < low:
            high, low = low, high


        cdef double median = (high + low) * 0.5


        cdef double retval = self.oscillator_2.update(median) - self.oscillator_1.update(median)


        if self.counter <= self.window:
            return np.nan
        return retval

    cpdef batch(self, high, low):
        mid = (high + low) * 0.5
        return self.oscillator_1.batch(mid) - self.oscillator_2.batch(mid)


cdef class Delay:
    cdef int index
    cdef int max
    cdef bint fillna
    cdef double[:] buffer_view
    cdef object buffer
    cdef bint first
    
    def __init__(self, int window=1, bint fillna=True):
        self.index = 0
        self.max = window
        self.fillna = fillna
        self.buffer = np.zeros(self.max)
        self.buffer_view = self.buffer
        self.first = True

    cpdef double update(self, double value):
        cdef int idx
        if self.first:
            self.first = False
            self.buffer[:] = value if self.fillna else np.nan
        retval = self.buffer_view[self.index]
        self.buffer_view[self.index] = value
        self.index += 1 
        if self.index  == self.max:
            self.index = 0
        return retval

        

cdef class EMA:
    """SMA - Simple Moving Average

    Args:
      window(int): n period
      fillna(bool): if True, fill nan values
    """

    cdef bint first_pass
    cdef long index
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
            self.first_pass = False

        if self.first_pass and not self.fillna:
            return np.nan

        return self.last_value

    cpdef batch(self, input):
        cdef long i
        cdef long j
        cdef double l = self.last_value
        retval = np.empty(input.shape[0], dtype=np.double)

        cdef double[:] input_view = input
        cdef double[:] output_view = retval

        if self.fillna:
            for i in range(j):
                output_view[i] = l = self.weighted_multiplier * input_view[i] + l * self.inverted_multiplier
            self.last_value = l
            self.index += input.shape[0]
        else:
            for i in range(j):
                l = self.weighted_multiplier * input_view[i] + l * self.inverted_multiplier
                self.index += 1
                if self.index < self.window:
                    output_view[i] = np.nan
                else:
                    output_view[i] = l
            self.last_value = l
        return retval


cdef class KamaIndicator():
    """KamaIndicator - Kaufman's Adaptive Moving Average (KAMA),
    created by Perry Kaufman, is an advanced moving average that
    responds to both trends and volatility. It is a potent
    trend-following indicator based on the Exponential Moving Average
    (EMA). The KAMA closely follows the price when noise levels are
    low, and it smooths out the noise when the price fluctuates. KAMA
    can be used like other moving averages to visualize the trend, and
    price crossing it may indicate a change in
    direction. Additionally, price can bounce off the KAMA, which can
    act as dynamic support and resistance. KAMA is often combined with
    other signals and analysis techniques.

    Args:
      window: (10) n period
      pow1: (2) number of periods for the fastest EMA constant
      pow2: (30) number of periods for the slowest EMA constant
      fillna: fill in nan values
    """
    pass

    
cdef class MACD():
    """MCAD - Moving average convergence divergence.
    https://en.wikipedia.org/wiki/MACD

    MACD(a, b, c)
    
    Args:
      a, b, c: The EMA period parameters
    """

    cdef EMA a
    cdef EMA b
    cdef EMA c
    cdef long counter
    cdef bint fillna
    cdef int window    

    def __init__(self, int a=12, int b=26, int c=9, bint fillna=False):
        """
        https://school.stockcharts.com/doku.php?id=technical_indicators:moving_average_convergence_divergence_macd
        """
        self.a = EMA(window=a, fillna=True)
        self.b = EMA(window=b, fillna=True)
        self.c = EMA(window=c, fillna=fillna)
        self.counter = 0
        self.fillna = fillna
        self.window = max(a,b) +c 

    cpdef double update(self, double value):
        cdef double retval
        try:
            retval = self.c.update(self.a.update(value) - self.b.update(value))
            if not self.fillna and self.counter < self.window:
                return np.nan
            return retval
        finally:
            self.counter += 1 

    cpdef batch(self, input):
        cdef long i 
        retval = np.empty(input.shape[0], dtype=np.double)

        cdef double[:] input_view = input
        cdef double[:] output_view = retval

        for i in range(input.shape[0]):
            output_view[i] = self.update(input_view[i])
        return retval

            
cdef class MassIndex():
    cdef EMA single
    cdef EMA double
    cdef Summation summation
    cdef long counter
    cdef int window
    cdef bint fillna
    
    def __init__(self, int single=9, int double=9, int summation=25, bint fillna=False):
        self.single = EMA(window=single, fillna=True)
        self.double = EMA(window=double, fillna=True)
        self.summation = Summation(window=summation, fillna=True)
        self.counter = 0
        self.window = max(single, double) + summation
        self.fillna = fillna

    cpdef double update(self, double high, double low):
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

        

cdef class SMA():
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
    cpdef double update(self, double value):
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

    cpdef batch(self, input):
        cdef long i 
        retval = np.empty(input.shape[0], dtype=np.double)

        cdef double[:] input_view = input
        cdef double[:] output_view = retval

        for i in range(input.shape[0]):
            output_view[i] = self.update(input_view[i])
        return retval



cdef struct PercentagePriceOscillatorResponse:
    double ppo
    double signal
    double histogram
    

cdef class PercentagePriceOscillator():
    cdef EMA oscillator_1
    cdef EMA oscillator_2
    cdef EMA oscillator_3
    cdef int window
    cdef long counter
    
    def __init__(self, int window_1=12, int window_2=26, int window_3=9, fillna=False):
        self.oscillator_1 = EMA(window=window_1, fillna=True)
        self.oscillator_2 = EMA(window=window_2, fillna=True)
        self.oscillator_3 = EMA(window=window_3, fillna=True)
        self.window = max(window_1, window_2, window_3) if not fillna else 0
        self.counter = 0

    cpdef PercentagePriceOscillatorResponse update(self, double close):
        self.counter += 1

        cdef double o26 = self.oscillator_2.update(close)

        retval = PercentagePriceOscillatorResponse()
        retval.ppo = ((self.oscillator_1.update(close) - o26)  / o26) * 100
        retval.signal = self.oscillator_3.update(retval.ppo)
        retval.histogram = retval.ppo - retval.signal

        return retval

    cpdef batch(self, close):
        o26 = self.oscillator_2.batch(close)
        ppo = ((self.oscillator_1.batch(close) - o26) / o26 ) * 100
        signal = self.oscillator_3.batch(ppo)
        histogram = ppo - signal
        return {'histogram':histogram, 'signal': signal, 'ppo': ppo}
        

        

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
    cpdef double update(self, double value):
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

    cpdef batch(self, input):
        cdef long i 
        retval = np.empty(input.shape[0], dtype=np.double)

        cdef double[:] input_view = input
        cdef double[:] output_view = retval

        for i in range(input.shape[0]):
            output_view[i] = self.update(input_view[i])
        return retval




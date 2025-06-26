import sys
import numpy as np
cimport numpy as cnp
cimport cython
cnp.import_array()
from collections import namedtuple
from cython.cimports.libc.math import sqrt

cdef class ATR:
    cdef Delay yesterday_close
    cdef Delay last_tr
    cdef double window

    def __init__(self, float window = 14, bint fillna=True):
        self.window = window
        self.yesterday_close = Delay(1)
        self.last_tr = Delay(1)

    cpdef double update(self, double close, double high, double low):
        cdef prev_close = self.yesterday_close.update(close)
        cdef double tr = max(
            high - low,
            abs(prev_close - high),
            abs(prev_close-low))
        cdef double prior_tr = self.last_tr.update(tr)
        return (prior_tr * (self.window-1) + tr) / self.window

cdef class ATRP:
    cdef ATR atr

    def __init__(self, float window=14, bint fillna=True):
        self.atr = ATR(window, fillna)

    cpdef double update(self, double close, double high, double low):
        return self.atr.update(close, high, low) / close


cdef class AwesomeOscillator:
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
            self.buffer[:] = 0 if self.fillna else np.nan
        retval = self.buffer_view[self.index]
        self.buffer_view[self.index] = value
        self.index += 1 
        if self.index  == self.max:
            self.index = 0
        return retval

    cpdef double peek(self):
        return self.buffer[self.index]

        

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
    
    def __init__(self, double window, bint fillna=False):
        self.first_pass = True
        self.index = 0
        self.window = max(int(window), 1)
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
        cdef long j = input.shape[0]
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

cdef class Kama():
    """
    Args:
      window: (10) n period
      pow1: (2) number of periods for the fastest EMA constant
      pow2: (30) number of periods for the slowest EMA constant
      fillna: fill in nan values
    """

    cdef Delay vol
    cdef Delay window
    cdef Summation den
    cdef double _pow1
    cdef double _pow2
    cdef bint first
    cdef Delay kama

    def __init__(self, int window=10, int fast_ema=2, int slow_ema=30, fillna=True):
        
        self.vol = Delay(1)
        self.window = Delay(window)
        self.den = Summation(window)
        self._pow1 = fast_ema
        self._pow2 = slow_ema
        self.first = True
        self.kama = Delay(1)

    cpdef double update(self, double close):
        cdef double vol = abs(close - self.vol.update(close))
        cdef double er_num = abs(close - self.window.update(close))
        cdef double er_den = self.den.update(vol)

        cdef double efficiency_ratio = 0 if er_den == 0 else er_num / er_den
        cdef double smoothing_constant = (
            (
                efficiency_ratio * (2.0 / (self._pow1 + 1) - 2.0 / (self._pow2 + 1.0))
                + 2 / (self._pow2 + 1.0)
            )
            ** 2.0)

        if np.isnan(smoothing_constant):
            return np.nan
        elif self.first:
            self.kama.update(close)
            self.first = False 
            return close
        cdef double peek = self.kama.peek()
        cdef double retval = peek + smoothing_constant * (close - peek)
        self.kama.update(retval)
        return retval

    cpdef batch(self, input):
        cdef long i 
        retval = np.empty(input.shape[0], dtype=np.double)

        cdef double[:] input_view = input
        cdef double[:] output_view = retval

        for i in range(input.shape[0]):
            output_view[i] = self.update(input_view[i])
        return retval

    

    
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



cdef struct PercentagePriceResponse:
    double ppo
    double signal
    double histogram
    

cdef class PercentagePrice():
    cdef EMA oscillator_1
    cdef EMA oscillator_2
    cdef EMA oscillator_3
    cdef int window
    cdef long counter
    
    def __init__(self, double window_1=12, double window_2=26, double window_3=9, fillna=False):
        self.oscillator_1 = EMA(window=window_1, fillna=True)
        self.oscillator_2 = EMA(window=window_2, fillna=True)
        self.oscillator_3 = EMA(window=window_3, fillna=True)
        self.window = int(max(window_1, window_2, window_3)) if not fillna else 0
        self.counter = 0

    cpdef PercentagePriceResponse update(self, double close):
        self.counter += 1

        cdef double o26 = self.oscillator_2.update(close)

        retval = PercentagePriceResponse()
        retval.ppo = ((self.oscillator_1.update(close) - o26)  / o26) * 100
        retval.signal = self.oscillator_3.update(retval.ppo)
        retval.histogram = retval.ppo - retval.signal

        return retval

    cpdef batch(self, close):
        cdef int j = close.shape[0]
        cdef int i
        ppo = np.empty(input.shape[0], dtype=np.double)
        signal = np.empty(input.shape[0], dtype=np.double)
        histogram = np.empty(input.shape[0], dtype=np.double)

        cdef double[:] input_view=close
        cdef double[:] ppo_view=ppo
        cdef double[:] signal_view=signal
        cdef double[:] histogram_view=histogram
        
        for i in range(j):
            output = self.update(input_view[i])
            ppo_view[i] = output.ppo
            signal_view[i] = output.signal
            histogram_view[i] = output.histogram

        return {'ppo': ppo, 'signal': signal,
                'histogram':histogram}


cdef struct PercentageVolumeResponse:
    double pvo
    double signal
    double histogram

    
cdef class PercentageVolume:
    """PercentageVolume: The Percentage Volume Oscillator (PVO) is a
    momentum indicator for volume. It calculates the difference
    between two volume-based moving averages as a percentage of the
    larger moving average.  It is similar to the MACD and the
    Percentage Price Oscillator (PPO).
    
    https://school.stockcharts.com/doku.php?id=technical_indicators:percentage_volume_oscillator_pvo
    """

    cdef EMA oscillator_1
    cdef EMA oscillator_2
    cdef EMA signal
    cdef long counter

    def __init__(self, double window_1=12, double window_2=26, double signal=9, fillna=True):
        self.oscillator_1 = EMA(window=window_1, fillna=True)
        self.oscillator_2 = EMA(window=window_2, fillna=True)
        self.signal = EMA(window=signal, fillna=True)

        if fillna:
            self.counter = 0
        else:
            self.counter = -int(max(window_1, window_2))

    cpdef PercentageVolumeResponse update(self, double volume):
        self.counter += 1

        cdef double ema_2 = self.oscillator_2.update(volume)

        cdef double pvo = 100 * (self.oscillator_1.update(volume) - ema_2) / (ema_2)
        cdef double signal = self.signal.update(pvo)
        cdef double histogram = pvo - signal
        
        cdef PercentageVolumeResponse retval

        if self.counter < 0:
            retval.pvo = np.nan
            retval.signal = np.nan
            retval.histogram = np.nan
        else:
            retval.pvo = pvo
            retval.signal = signal
            retval.histogram = histogram

        return retval

    cpdef batch(self, volume):
        cdef int j = volume.shape[0]
        cdef int i

        pvos = np.empty(j, dtype=np.double)
        signals = np.empty(j, dtype=np.double)
        histograms = np.empty(j, dtype=np.double)

        cdef double[:] input_view=volume
        cdef double[:] pvo_view=pvos
        cdef double[:] signal_view=signals
        cdef double[:] histogram_view=histograms

        cdef double pvo = 0
        cdef double signal = 0
        cdef double histogram = 0
        cdef double ema_2 = 0
        
        for i in range(j):
            self.counter += 1
            ema_2 = self.oscillator_2.update(input_view[i])
            pvo = 100 * (self.oscillator_1.update(input_view[i]) - ema_2) / ema_2
            signal = self.signal.update(pvo)

            if self.counter < 0:
                pvo_view[i] = np.nan
                signal_view[i] = np.nan
                histogram_view[i] = np.nan
            else:
                pvo_view[i] = pvo
                signal_view[i] = signal
                histogram_view[i] = pvo - signal

        return {'pvo': pvos, 'signal': signals,
                'histogram':histograms}


cdef class ROC:
    """ROC:

    Args:
      window: Window size
      fillna:
    """

    cdef Delay close
    cdef int window
    cdef bint fillna
    cdef long counter 

    def __init__(self, int window, bint fillna=True):
        self.close = Delay(window=window, fillna=True)
        self.window = window
        self.fillna = fillna
        self.counter = 0
        
    cpdef double update(self, double close):
        cdef double close_ago = self.close.update(close)
        if not self.fillna and self.counter < self.window:
            self.counter += 1
            return np.nan
        if close_ago == 0:
            self.counter += 1
            return 0
        self.counter += 1
        return  ((close - close_ago) / (close_ago)) * 100

    cpdef batch(self, close):
        cdef int j = close.shape[0]
        cdef int i
        cdef double close_ago
        
        retval = np.empty(close.shape[0])
        cdef double[:] input_view = close
        cdef double[:] output_view = retval
        
        for i in range(j):
            close_ago = self.close.update(input_view[i])
            if not self.fillna and self.counter < self.window:
                output_view[i] = np.nan
            elif close_ago == 0:
                output_view[i] = 0
            else:
                output_view[i] = 100 * ((input_view[i] - close_ago) / (close_ago))
            self.counter += 1

        return retval
        
cdef class RSI:
    """RSI: The Relative Strength Index (RSI)
    """

    cdef bint fillna
    cdef int window
    cdef long counter
    cdef double prev
    cdef double high
    cdef double low

    def __init__(self, int window=14, bint fillna=True):
        self.fillna = fillna
        self.window = window
        self.counter = 0
        self.prev = 0
        self.high = 0
        self.low = 0 

    cpdef double update(self, double value):
        try:
            if self.counter == 0:
                self.prev = value
                if self.fillna:
                    return 50
                else:
                    return np.nan
            elif self.counter <= self.window:
                if value < self.prev:
                    self.low = (self.low * (self.counter-1) +  self.prev - value) / self.counter
                elif value > self.prev:
                    self.high = (self.high * (self.counter-1) + value - self.prev) / self.counter
                if not self.fillna:
                    return np.nan
                elif self.low == 0:
                    return 100
                else:
                    return 100 - (100 / (1 + (self.high / self.window) / (self.low / self.window)))
            else:
                if value < self.prev:
                    self.low = (self.low * (self.window - 1) + self.prev - value) / self.window
                elif self.low == 0:
                    return 100
                elif value > self.prev:
                    self.high = (self.high * (self.window-1) + value - self.prev) / self.window
                return 100 - (100 / (1 + (self.high / self.window) / (self.low / self.window)))
        finally:
            self.prev = value
            self.counter += 1
    

cdef class Summation:
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
    
    def __init__(self, int window, bint fillna=True):
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


cdef class High:
    """High - Return the highest value in a window.

    Args:
      window(int): n period
      fillna(bool): if True, fill nan values
    """

    cdef bint first_pass
    cdef int window_size
    cdef int offset
    cdef object window
    cdef double[:] window_view
    cdef bint fillna

    def __init__(self, int window, bint fillna=True):
        self.window = np.zeros(window)
        self.window_view = self.window
        self.first_pass = True
        self.offset = 0
        self.window_size = window
        self.fillna = fillna

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef double update(self, double value):
        cdef double max
        cdef int i
        
        self.window_view[self.offset] = value

        if self.first_pass:
            if not self.fillna:
                self.offset += 1
                if self.offset >= self.window_size:
                    self.first_pass = False
                    self.offset = 0 
                return np.nan
            max = self.window_view[0]
            for i in range(1, self.offset+1):
                if self.window_view[i] > max:
                    max = self.window_view[i]
            self.offset += 1
            if self.offset >= self.window_size:
                self.first_pass = False
                self.offset = 0 
            return max
        max = self.window_view[0]
        for i in range(1, self.window_size):
            if self.window_view[i] > max:
                max = self.window_view[i]
        self.offset += 1
        if self.offset >= self.window_size:
            self.offset = 0
        return max

    
cdef class Low:
    """Low - Return the lowest value in a window.

    Args:
      window(int): n period
      fillna(bool): if True, fill nan values
    """

    cdef bint first_pass
    cdef int window_size
    cdef int offset
    cdef object window
    cdef double[:] window_view
    cdef bint fillna

    def __init__(self, int window, bint fillna=True):
        self.window = np.zeros(window)
        self.window_view = self.window
        self.first_pass = True
        self.offset = 0
        self.window_size = window
        self.fillna = fillna

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef double update(self, double value):
        cdef double min
        cdef int i
        
        self.window_view[self.offset] = value

        if self.first_pass:
            if not self.fillna:
                self.offset += 1
                if self.offset >= self.window_size:
                    self.first_pass = False
                    self.offset = 0 
                return np.nan
            min = self.window_view[0]
            for i in range(1, self.offset+1):
                if self.window_view[i] < min:
                    min = self.window_view[i]
            self.offset += 1
            if self.offset >= self.window_size:
                self.first_pass = False
                self.offset = 0 
            return min
        min = self.window_view[0]
        for i in range(1, self.window_size):
            if self.window_view[i] < min:
                min = self.window_view[i]
        self.offset += 1
        if self.offset >= self.window_size:
            self.offset = 0
        return min
            

cdef class StochRSI:
    """StochRSI: Measures the RSI relative to its high/low range.

    https://school.stockcharts.com/doku.php?id=technical_indicators:stochrsi

    Args:
      window: default 14
      fillna: default True
    """

    cdef RSI rsi
    cdef Low lowest_rsi
    cdef High highest_rsi
    cdef int window_size
    cdef bint fillna

    def __init__(self, int window=14, bint fillna=True):
        self.rsi = RSI(window, True)
        self.lowest_rsi = Low(window, fillna)
        self.highest_rsi = High(window, fillna)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef double update(self, double value):
        cdef double rsi = self.rsi.update(value)
        cdef double low = self.lowest_rsi.update(rsi)
        cdef double high = self.highest_rsi.update(rsi)
        if high == low:
            return 0
        return (rsi / low ) / (high - low)

cdef class TSI:
    """TSI: True Strength Index

    https://school.stockcharts.com/doku.php?id=technical_indicators:true_strength_index
    """

    cdef Delay P
    cdef EMA A
    cdef EMA B
    cdef Delay D
    cdef EMA E
    cdef EMA F

    def __init__(self, int window_1 = 25, int window_2 = 13):

        self.P = Delay(1)
        self.A = EMA(window_1, True)
        self.B = EMA(window_2, True)
        self.D = Delay(1)
        self.E = EMA(window_1, True)
        self.F = EMA(window_2, True)

    def update(self, double x):
        return (100 * self.B.update(self.A.update(x-self.P.update(x)))) / (self.F.update(self.E.update(abs(x-self.D.update(x)))))


cdef class StdDev:
    """StdDev: Standard Deviation

    https://www.khanacademy.org/math/statistics-probability/summarizing-quantitative-data/variance-standard-deviation-population/a/calculating-standard-deviation-step-by-step
    """

    cdef object history
    cdef double[:] history_view
    cdef bint fillna
    cdef int offset
    cdef int window_size
    cdef bint first_pass
    cdef SMA mean

    def __init__(self, int window, bint fillna=True):
        self.history = np.zeros(window)
        self.history_view = self.history
        self.first_pass = True
        self.window_size = window
        self.offset = 0
        self.mean = SMA(window, True)
        self.fillna = fillna

    @cython.boundscheck(False) # turn off bounds-checking for entire function
    @cython.wraparound(False)
    cpdef double update(self, double value):
        cdef int i
        cdef double tally
        cdef double mean

        try:
            self.history_view[self.offset] = value
            self.offset += 1
            mean = self.mean.update(value)
            tally = 0
            if self.first_pass:
                if not self.fillna:
                    return np.nan
                for i in range(self.offset):
                    tally += (self.history_view[i] - mean) ** 2
                tally /= self.offset
            else:
                for i in range(self.window_size):
                    tally += (self.history_view[i] - mean) ** 2
                tally /= self.window_size
            return sqrt(tally)
            
        finally:
            if self.offset >= self.window_size:
                self.offset = 0
                self.first_pass = False
                


cdef struct BollingerBandResponse:
    double middle
    double lower
    double upper
    
    
cdef class BollingerBands:
    """BollingerBands: Bollinger Bands
    
    https://school.stockcharts.com/doku.php?id=technical_indicators:bollinger_bands 
    """

    cdef SMA sma
    cdef StdDev stddev

    def __init__(self, int window=20, bint fillna=True):
        self.sma = SMA(window, fillna=fillna)
        self.stddev=StdDev(window, fillna=fillna)

    @cython.boundscheck(False) # turn off bounds-checking for entire function
    @cython.wraparound(False)
    cpdef BollingerBandResponse update(self, double value):

        cdef BollingerBandResponse retval

        cdef double stddev = self.stddev.update(value)
        retval.middle = self.sma.update(value)
        
        retval.upper = retval.middle + stddev * 2
        retval.lower = retval.middle - stddev * 2
        
        return retval
        

    
        

    

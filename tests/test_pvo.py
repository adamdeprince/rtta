import math
from unittest import TestCase, main

import numpy as np
import pandas as pd

from rtta.indicator import PercentageVolume
from ta.momentum import PercentageVolumeOscillator as PercentageVolumeOscillator

class PercentageVolumeTest(TestCase):

    def test_against_reference(self):
        data = np.random.rand(100)
        pvo = PercentageVolume(fillna=True)
        for (x,y) in zip(PercentageVolumeOscillator(fillna=True, volume=pd.Series(data)).pvo(),
                         [pvo.update(x)['pvo'] for x in data]):
            self.assertAlmostEqual(x,y)
        pvo = PercentageVolume(fillna=True)
        for (x,y) in zip(PercentageVolumeOscillator(fillna=True, volume=pd.Series(data)).pvo_signal(),
                         [pvo.update(x)['signal'] for x in data]):
            self.assertAlmostEqual(x,y)
        pvo = PercentageVolume(fillna=True)
        for (x,y) in zip(PercentageVolumeOscillator(fillna=True, volume=pd.Series(data)).pvo_hist(),
                         [pvo.update(x)['histogram'] for x in data]):
            self.assertAlmostEqual(x,y)

    def test_against_reference_bulk(self):
        data = np.random.rand(100)
        pvo = PercentageVolume(fillna=True)
        for (x,y) in zip(PercentageVolumeOscillator(fillna=True, volume=pd.Series(data)).pvo(),
                         pvo.batch(data)['pvo']):
            self.assertAlmostEqual(x,y)
        pvo = PercentageVolume(fillna=True)
        for (x,y) in zip(PercentageVolumeOscillator(fillna=True, volume=pd.Series(data)).pvo_signal(),
                         pvo.batch(data)['signal']):
            self.assertAlmostEqual(x,y)
        pvo = PercentageVolume(fillna=True)
        for (x,y) in zip(PercentageVolumeOscillator(fillna=True, volume=pd.Series(data)).pvo_hist(),
                         pvo.batch(data)['histogram']):
            self.assertAlmostEqual(x,y)
        
        


if __name__ == "__main__":
    main()

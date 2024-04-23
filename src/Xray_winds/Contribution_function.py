import ChiantiPy.core as ch
import ChiantiPy.tools.filters as chfilter
import numpy as np


def create_contribution_function(wavelength_range, min_abund=1e-5, density=1.e+10):
    wvl = np.linspace(wavelength_range.min(), wavelength_range.max(), 3001)
    temperature = np.logspace(4,8, 201)
    s = ch.spectrum(temperature, density, wvl, filter=(chfilter.gaussian, 0.1),
                    em =1,
                    doContinuum=True,
                    minAbund=min_abund,
                    verbose=False)
    return wvl, temperature, s
    
    


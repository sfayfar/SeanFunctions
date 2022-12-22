import numpy as np
import lmfit as lm


def fitPeak(data,xleft,xright,peakType='Gaussian',constant=False):
    '''
    Fit data to a peak model with the provides bounds and the addition of a constant background term
    
    Parameters
    --------
    data : array_like
        Input 2D array containing the data to fit.
    xleft : float
        Left boundary of the data to fit.
    xright : float
        Right boundary of the data to fit.
    peakType : str, optional
        The type of peak to use for the fit. The default choice is a Gaussian model.
        All models are listed below.
    constant : bool, optional
        Whether to add a constant background value to the fit.


    Returns
    --------
    fitResult : lmfit ModelResult object
        The ModelResult contains the fitting results created by lmfit. 
    '''
    
    if peakType not in lm.models.lmfit_models.keys():
        print('Peak type not available. Options are:')
        for keys in lm.models.lmfit_models.keys():
            print(keys)
        return -1
    
    datax = data[:,0]
    datay = data[:,1]
    
    modPeak = lm.models.lmfit_models[peakType]()
    modConst = lm.models.ConstantModel()
    if constant:
        modCombined = modPeak + modConst
    else:
        modCombined = modPeak
    
    dataLocations = np.where((datax >= xleft) & (datax <= xright))[0]
    
    fitx = datax[dataLocations]
    fity = datay[dataLocations]
    
    
    params = modConst.guess(fity,x=fitx)
    params+= modPeak.guess(fity,x=fitx)
    
    params['c'].set(min=0,max=np.max(fity))
    
    fitResult = modCombined.fit(fity,params,x=fitx)
    
    return fitResult
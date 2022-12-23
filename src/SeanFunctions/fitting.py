import numpy as np
import lmfit as lm


def fitPeak(data,xleft,xright,peakType='Gaussian',constant=False):
    '''
    Fit data to a peak model with the provided bounds and the addition of a constant background term
    
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


def find_max(fitfunction,bounds):
    '''
    Finds the maximum value of a Gaussian shaped function determined through a fit

    Parameters
    ---------
    fitfunction : lmfit fitted class
        Fitting result from lmfit
    
    bounds : array_like
        A list of the bounds where the peak is located.

    Return
    --------
    out : array with xvalue, maxvalue, maxvalueunc


    Example
    --------
    modGaus = lmfit.models.Gaussian()
    result = modGaus.fit(y,params,x=x)
    maximum = find_max(result,[1,2])
        '''
    from scipy.optimize import minimize_scalar
    
    def function(xvalue):
        return -fitfunction.eval(x=xvalue)
    xvalue = minimize_scalar(function,bounds=bounds,method='bounded').x
    maxvalue = fitfunction.eval(x=xvalue)
    maxvalueunc = fitfunction.eval_uncertainty(x=xvalue)[0]
    out = np.array([xvalue, maxvalue, maxvalueunc])
    return out


def convertToUncFloat(paramResult):
    '''
    Converts a parameter output from lmfit to a uncertainties ufloat.

    Parameters
    ------
    paramResult : Paramtere object from lmfit
            Input the parameter after a fit to retrive the value
            and uncertainty into a ufloat.

    Returns
    ------
    out : ufloat
        Returns a uncertainties ufloat value.
    '''
    from uncertainties import ufloat

    out = ufloat(paramResult.value,paramResult.stderr)

    return out
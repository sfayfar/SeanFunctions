import lmfit as lm
import numpy as np


def fitPeak(
    data,
    xleft,
    xright,
    weight=True,
    peakType="Gaussian",
    constant=False,
    ampParams=None,
    centParams=None,
    constParams=None,
    gammaParams=None,
    skewParams=None,
):
    """
    Fit data to a peak model with the provided bounds and the addition of a constant background term

    Parameters
    --------
    data : array_like
        Input 2D array containing the data to fit.

    xleft : float
        Left boundary of the data to fit.

    xright : float
        Right boundary of the data to fit.

    weight : bool, or array
        True/False whether to use third data column to weight the fitting
        Or input array to use to weight the fitting as 1/yerr

    peakType : str, optional
        The type of peak to use for the fit. The default choice is a Gaussian model.
        All models are listed below.

    constant : bool, optional
        Whether to add a constant background value to the fit.

    ampParams : dict, optional
        Dictionary of optional keyword args for the amplitude param.

    centParams : dict, optional
        Dictionary of optional keyword args for the center param.

    gammaParams : dict, optional
        Dictionary of optional keyword args for the gamma param.

    constParams : dict, optional
        Dictionary of optional keyword args for the constant param

    skewParams : dict, optional
        Dictionary of optional keyword args for the skew param


    Returns
    --------
    fitResult : lmfit ModelResult object
        The ModelResult contains the fitting results created by lmfit.



    Models
    --------
    Constant
    Complex Constant
    Linear
    Quadratic
    Polynomial
    Spline
    Gaussian
    Gaussian-2D
    Lorentzian
    Split-Lorentzian
    Voigt
    PseudoVoigt
    Moffat
    Pearson4
    Pearson7
    StudentsT
    Breit-Wigner
    Log-Normal
    Damped Oscillator
    Damped Harmonic Oscillator
    Exponential Gaussian
    Skewed Gaussian
    Skewed Voigt
    Thermal Distribution
    Doniach
    Power Law
    Exponential
    Step
    Rectangle
    Expression

    """

    if peakType not in lm.models.lmfit_models.keys():
        keyList = []
        for keys in lm.models.lmfit_models.keys():
            keyList.append(keys)
        raise ValueError("Peak types:", " ".join(keyList))

    datax = data[:, 0]
    datay = data[:, 1]
    if isinstance(weight,bool):
        if (data.shape[1] == 3) and weight:
            dataErr = data[:,2]
        else:
            dataErr = None
    elif isinstance(weight,(list,tuple,np.ndarray)):
        dataErr = weight
    else:
        dataErr = None

    modPeak = lm.models.lmfit_models[peakType]()
    modConst = lm.models.ConstantModel()
    if constant:
        modCombined = modPeak + modConst
    else:
        modCombined = modPeak

    dataLocations = np.where((datax >= xleft) & (datax <= xright))[0]

    fitx = datax[dataLocations]
    fity = datay[dataLocations]
    

    params = modPeak.guess(fity, x=fitx)
    if constant:
        params += modConst.guess(fity, x=fitx)
        if constParams is not None:
            params["c"].set(**constParams)
        else:
            params["c"].set(min=0, max=np.max(fity))

    if ampParams is not None:
        params["amplitude"].set(**ampParams)

    if centParams is not None:
        params["center"].set(**centParams)

    if gammaParams is not None:
        params["gamma"].set(**gammaParams)

    if skewParams is not None:
        params["skew"].set(**skewParams)

    kwargs = {}
    if dataErr is not None:
        kwargs['weight'] = 1/dataErr

    fitResult = modCombined.fit(fity, params, x=fitx,**kwargs)

    return fitResult


def find_max(fitfunction, bounds, min=False, evalUnc=True, params=None):
    """
    Finds the maximum value of a Gaussian shaped function determined through a fit

    Parameters
    ---------
    fitfunction : lmfit fitted class
        Fitting result from lmfit

    bounds : array_like
        A list of the bounds where the peak is located.

    min : bool, optional
        Optional keyword for finding minimum rather than maximum

    Return
    --------
    out : array with xvalue, maxvalue, maxvalueunc


    Example
    --------
    modGaus = lmfit.models.Gaussian()
    result = modGaus.fit(y,params,x=x)
    maximum = find_max(result,[1,2])
    """
    from scipy.optimize import minimize_scalar

    if min:
        scale = 1
    else:
        scale = -1

    def function(xvalue):
        return scale * fitfunction.eval(x=xvalue, params=params)

    xvalue = minimize_scalar(function, bounds=bounds, method="bounded").x
    maxvalue = fitfunction.eval(x=xvalue, params=params)
    if evalUnc:
        maxvalueunc = fitfunction.eval_uncertainty(x=xvalue, params=params)[0]
        out = np.array([xvalue, maxvalue, maxvalueunc])
    else:
        out = np.array([xvalue, maxvalue])
    return out


def convertToUncFloat(paramResult):
    """
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
    """
    from uncertainties import ufloat

    out = ufloat(paramResult.value, paramResult.stderr)

    return out

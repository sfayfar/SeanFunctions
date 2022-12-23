# from numpy import array, sqrt, asarray, argmin, abs, arange, zeros, sum, sinc, pi, max, column_stack
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.integrate import simpson
import os
from pathlib import Path
# from varname import nameof


gr = (1 + np.sqrt(5))/2 #Defines the golden ratio



def find_nearest(array, value):
    '''
    Finds the nearest value in an array and its position

    Input
    -----------
    array, value


    Return 
    ----------
    Nearest value
    '''
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]


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
    def function(xvalue):
        return -fitfunction.eval(x=xvalue)
    xvalue = minimize_scalar(function,bounds=bounds,method='bounded').x
    maxvalue = fitfunction.eval(x=xvalue)
    maxvalueunc = fitfunction.eval_uncertainty(x=xvalue)[0]
    out = np.array([xvalue, maxvalue, maxvalueunc])
    return out




def discrete_integral(data):
    '''
    Calculate a continuous discrete integral of data using the simpson method

    Parameters
    ----------
    data : array_like
        The input array to be integrated.

    Returns
    -------
    dataInt : float value of integration

    '''
    dataInt = data * 0 
    for i in range(1,len(data)+1):
        dataInt[i-1,:] = np.array([data[i-1,0],simpson(data[:i,1],data[:i,0])])
    return dataInt



def fourierbesseltransform(q,int1,unpack=None):
    '''
    Discrete fourier bessel transform for conversion between S(Q) and g(r)

    Parameters
    --------
    q : array_like
        Input array of Q (or r) values
    int1 : array_like
        Input array of the structure factor S values (or PDF values)
    unpack : bool, optional
        If unpack is True, the result will be output as a tuple to 
        more easily define separate variables from the result

    Returns 
    --------
    r : ndarray
        array of the real space distance (or Q) values.
    ft : ndarray
        fourier transform result. Returns the PDF (or S) values.
    '''    
    #First define delta Q
    dq = q[1]-q[0] #Q or r is assumed to be equidistant
    
    q2 = q**2
    
    intq2 = int1 * q2 #Multiply by Q^2 before
    
    nq = len(q)
    
    r = np.arange(nq) * np.pi / np.max(q) #Define the values of r
    nr = len(r)
    ft = np.zeros(nr) #Create empty array (detault is float64)
    
    for i in range(nr):
        x = q * r[i]
        ft[i] = sum(intq2 * np.sinc(x/np.pi)) #np.sinc = sin(pi*x)/(pi*x)
    
    ft *= dq
    
    if unpack is None:
        return np.column_stack((r, ft))
    else:
        return r, ft
    
    
    
# def SaveToFile(fname,*data,header=None,directory=None,**kwargs):
    
#     #Function to save numpy arrays to a file
#     currDir = Path(os.getcwd())
#     if directory is None:
#         outputDir = currDir
#     else:
#         outputDir = Path(directory)
#         if not outputDir.exists():
#             response = input(f'The directory:\n{outputDir}\n does not exist. Should it be created? y/n')
#             if response == 'y':
#                 outputDir.mkdir(parents=True, exist_ok=True)
#                 print('Directory created.')
#             else:
#                 print('Function aborted.')
#                 return
#     if '.' not in fname:
#         fname += '.dat'
#     outputFile = outputDir / str(fname)

#     openFile = open(outputFile,'w')
#     if header is not None:
#         if '#' not in header[0]:
#             header = ['#' + s for s in header]
#         openFile.writelines(header)
    
#     np.savetxt(openFile,np.column_stack(data),**kwargs)
#     openFile.close()
#     print(f'The data has been successfully save as: \n {outputFile}')
#     return


def moving_average(x, w):
    '''
    Computes the moving (or running or rolling) average of input data.

    Parameters
    ---------
    x : array_like
        Input data to perform moving average on
    w : int
        The window (or box) size to be averaged

    Returns
    -------
    out : ndarray
        The resulting array after performing the moving average. 
        The array will have a length of n-w+1
    '''
    out = np.convolve(x, np.ones(w), 'valid') / w
    return out
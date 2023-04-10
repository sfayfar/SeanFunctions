# from numpy import array, sqrt, asarray, argmin, abs, arange, zeros, sum, sinc, pi, max, column_stack
import numpy as np
from scipy.integrate import simpson
import uncertainties as unc
# import os
# from pathlib import Path
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



def convertToUncFloat(paramResult):
    '''
    Returns a ufloat from the inputed lmfit parameter class

    Parameters 
    --------
    paramResult : lmfit parameter class
                  Fitting parameters after using lmfit to perform fits.

    Returns
    --------
    out : uncertainties float (ufloat)
          The float value containing the uncertainty calculated from the fitting procedure.
    '''
    out = unc.ufloat(paramResult.value,paramResult.stderr)
    return out



def bin_data(dataArray_x,dataArray_y,minValue,maxValue,dataPoints,unpack=False,method='mean',density=False,interpEmpty=False):
    '''
    Returns rebinned x and y arrays

    Parameters
    ---------
    dataArray_x : array_like
                  The x-values of the array to be rebinned
    dataArray_y : array_like
                  The y-values of the array to be rebinned
    minValue : value
               The lower bound x-value of the new rebinned data
    maxValue : value
               The upper bound x-value of the new rebinned data
    dataPoint : value
                The number of bins to combine the data into
    unpack : bool, optional
             If unpack is True, the result will be output as a tuple to 
             more easily define separate variables from the result
    method : str, optional
             Choose either "sum" or "mean" for the method of combining data
    density : bool, optional
              Choose True to divide the data by the bin size creating a density = occurance/bin width

    Returns
    --------
    binnedArray_x : ndarray
                    New x-value bins
    binnedArray_y : ndarray
                    New y-values after binning

    Example
    ---------
    x = np.array([0,1,2,3,4,5,6,7,8,9,10])
    y = np.array([5,6,3,4,6,8,9,6,3,4,5])

    output = bin_data(x,y,0,10,5)
    print(output)

    [[1.  4.5]
    [3.  5. ]
    [5.  8.5]
    [7.  4.5]
    [9.  4.5]]
    
    
    '''
    from uncertainties.unumpy import uarray
    from scipy.interpolate import interp1d

    if method == 'mean':
        func = np.mean
    elif method == 'sum':
        func = np.sum
    else:
        print('Method must be either "mean" or "sum".')
        return -1

    binWidths = np.linspace(minValue,maxValue,dataPoints+1)
    binnedArray_x = np.zeros(dataPoints)
    
    if dataArray_y.dtype == 'O':
        binnedArray_y = uarray(range(dataPoints),range(dataPoints)) * 0.0
    else:
        binnedArray_y = np.zeros(dataPoints)
    
    for index in range(dataPoints):
        left = binWidths[index]
        right = binWidths[index+1]
        binSize = right - left
        binnedArray_x[index] = np.mean([left,right])

        if density: 
            norm = binSize
        else:
            norm = 1
        
        locations = np.where((dataArray_x >= left) & (dataArray_x < right))[0]
        if len(locations) != 0:
            binnedArray_y[index] = func(dataArray_y[locations]) / norm
        else:
            if interpEmpty:
                locationsExt = np.where((dataArray_x >= (left-binSize)) & (dataArray_x < (right+binSize)))[0]
                if len(locationsExt) != 0:
                    binnedArray_y[index] = interp1d(dataArray_x[locationsExt],dataArray_y[locationsExt],bounds_error=False,fill_value=0)(binnedArray_x[index])
                else:
                    if dataArray_y.dtype == 'O':
                        binnedArray_y[index] = unc.ufloat(0.0,0.0)
                    else:
                        binnedArray_y[index] = 0.0
            else:
                if dataArray_y.dtype == 'O':
                    binnedArray_y[index] = unc.ufloat(0.0,0.0)
                else:
                    binnedArray_y[index] = 0.0
        
        
    if unpack:
        return binnedArray_x, binnedArray_y
    else:
        return np.column_stack((binnedArray_x, binnedArray_y))
    


    
def bin_ndarray(ndarray, new_shape, operation='sum'):
    """
    Bins an ndarray in all axes based on the target shape, by summing or
        averaging.

    Number of output dimensions must match number of input dimensions and 
        new axes must divide old ones.

    Example
    -------
    >>> m = np.arange(0,100,1).reshape((10,10))
    >>> n = bin_ndarray(m, new_shape=(5,5), operation='sum')
    >>> print(n)

    [[ 22  30  38  46  54]
     [102 110 118 126 134]
     [182 190 198 206 214]
     [262 270 278 286 294]
     [342 350 358 366 374]]

    """
    operation = operation.lower()
    if not operation in ['sum', 'mean']:
        raise ValueError("Operation not supported.")
    if ndarray.ndim != len(new_shape):
        raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape,
                                                           new_shape))
    compression_pairs = [(d, c//d) for d,c in zip(new_shape,
                                                  ndarray.shape)]
    flattened = [l for p in compression_pairs for l in p]
    ndarray = ndarray.reshape(flattened)
    for i in range(len(new_shape)):
        op = getattr(ndarray, operation)
        ndarray = op(-1*(i+1))
    return ndarray
# from numpy import array, sqrt, asarray, argmin, abs, arange, zeros, sum, sinc, pi, max, column_stack
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.integrate import simpson
import os
from pathlib import Path
# from varname import nameof


gr = (1 + np.sqrt(5))/2 #Defines the golden ratio



def find_nearest(array, value):
    #Finds the nearest value in an array and its position
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]


def find_max(fitfunction,bounds):
    #Finds the maximum value of a Gaussian shaped function determined through a fit
    def function(xvalue):
        return -fitfunction.eval(x=xvalue)
    xvalue = minimize_scalar(function,bounds=bounds,method='bounded').x
    maxvalue = fitfunction.eval(x=xvalue)
    maxvalueunc = fitfunction.eval_uncertainty(x=xvalue)[0]
    return xvalue, maxvalue, maxvalueunc




def discrete_integral(data):
    #Calculate a continuous discrete integral of data using the simpson method
    # dataNameStr = 'integrate_'+nameof(data)
    dataInt = data * 0 
    for i in range(1,len(data)+1):
        dataInt[i-1,:] = np.array([data[i-1,0],simpson(data[:i,1],data[:i,0])])
    return dataInt



def fourierbesseltransform(q,int1,unpack=None):
    #This function is definted to convert between S(Q) and g(r) using a discrete Fourier-Bessel transform
    #The meaning of Q and r can be reversed
    #This function is based off the IDL procedure created by Joerg Neuefeind at ORNL
    
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
    
    
    
def SaveToFile(fname,*data,header=None,directory=None,**kwargs):
    
    #Function to save numpy arrays to a file
    currDir = Path(os.getcwd())
    if directory is None:
        outputDir = currDir
    else:
        outputDir = Path(directory)
        if not outputDir.exists():
            response = input(f'The directory:\n{outputDir}\n does not exist. Should it be created? y/n')
            if response == 'y':
                outputDir.mkdir(parents=True, exist_ok=True)
                print('Directory created.')
            else:
                print('Function aborted.')
                return
    if '.' not in fname:
        fname += '.txt'
    outputFile = outputDir / str(fname)

    openFile = open(outputFile,'w')
    if header is not None:
        if '#' not in header[0]:
            header = ['#' + s for s in header]
        openFile.writelines(header)
    
    np.savetxt(openFile,np.column_stack(data),**kwargs)
    openFile.close()
    print(f'The data has been successfully save as: \n {outputFile}')
    return


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w
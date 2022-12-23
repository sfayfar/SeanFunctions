import pandas as pd
import numpy as np
# from importlib import resources
# import io
import pkg_resources

stream = pkg_resources.resource_stream(__name__,'Data/AtomicFormFactorConstants.csv')
aff_DF = pd.read_csv(stream)

def atomic_form_factor(atom,QList,inputConst=None):
    if atom not in aff_DF['element'].values:
        print(f'The atom \'{atom}\' is not available.\nThe available atoms are:\n','\t'.join(map(str,aff_DF['element'].values)))
    else:
        if inputConst is None:
            const_values = aff_DF[aff_DF['element'] == atom].iloc[:,2:-2].values[0]
        else:
            const_values = inputConst
        sumData = 0.0
        for i in range(0,len(const_values)-1,2):
            sumData += const_values[i] * np.exp(-const_values[i+1] * (QList/(4*np.pi))**2)
        sumData += const_values[-1]
        return sumData
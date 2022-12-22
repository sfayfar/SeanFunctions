import pandas as pd
import numpy as np


def atomic_form_factor(atom,QList,inputConst=None):
    if atom not in df_data['element'].values:
        print(f'The atom \'{atom}\' is not available.\nThe available atoms are:\n','\t'.join(map(str,df_data['element'].values)))
    else:
        if inputConst is None:
            const_values = df_data[df_data['element'] == atom].iloc[:,2:-2].values[0]
        else:
            const_values = inputConst
        sumData = 0.0
        for i in range(0,len(const_values)-1,2):
            sumData += const_values[i] * np.exp(-const_values[i+1] * (QList/(4*np.pi))**2)
        sumData += const_values[-1]
        return sumData
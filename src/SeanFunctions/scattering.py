import pandas as pd
import numpy as np
# from importlib import resources
# import io


def atomic_form_factor_constants():
    '''
    Outputs a DataFrame of the coefficients for the analytical approximation to the atomic form factors. 
    The coefficients were taken from the International Tables for Crystallography at:
    http://it.iucr.org/Cb/ch6o1v0001/

    Parameters
    -------
    nothing


    Returns 
    --------
    aff_DF : Pandas DataFrame
        contains the coefficients to use the analytical approximation of the atomic form factors. 
    '''
    from pkg_resources import resource_stream

    stream = resource_stream(__name__,'Data/AtomicFormFactorConstants.csv')
    aff_DF = pd.read_csv(stream)
    return aff_DF



def atomic_form_factor(atom,QList,inputCoeff=None):
    '''
    Returns the Q dependent atomic form factor for each of the elements and ions. 
    The coefficients were taken from the International Tables for Crystallography at:
    http://it.iucr.org/Cb/ch6o1v0001/

    Parameters
    -------
    atom : str
        Put the name of the atom (or ion) for the atomic form factor.
        Ions have their charge put directly after the atom in the form
        X#+/-
        For example: Pt4+ or H1-
    
    QList : array_like
        The array of Q values that the atomic form factor will be calculated for.

    inputCoeff : array_like, optional
        Optionally input the coefficients manually rather than use the values from the table.
    '''

    aff_DF = atomic_form_factor_constants()

    if atom not in aff_DF['element'].values:
        print(f'The atom \'{atom}\' is not available.\nThe available atoms are:\n','\t'.join(map(str,aff_DF['element'].values)))
    else:
        if inputCoeff is None:
            coeff_values = aff_DF[aff_DF['element'] == atom].iloc[:,2:-2].values[0]
        else:
            coeff_values = inputCoeff
        sumData = 0.0
        for i in range(0,len(coeff_values)-1,2):
            sumData += coeff_values[i] * np.exp(-coeff_values[i+1] * (QList/(4*np.pi))**2)
        sumData += coeff_values[-1]
        return sumData


def neutron_scattering_lengths(rawTable=False):
    '''
    Returns a DataFrame of all the neutron scattering lengths. 
    The neutron scattering lengths were taken from:
    https://www.nist.gov/ncnr/neutron-scattering-lengths-list

    Parameters
    --------
    rawTable : bool, options
        Set true to output the raw table imported from the above link.
        Otherwise it will import a table with symbols removed (errors +/- values etc.)
        and corrected data types.

    Returns
    -------
    nsl_DF : DataFrame
        Contains the neutron scattering lengths of all elements and isotopes.

    Column    Unit    Quantity
    1         ---     Isotope
    2         ---     Natural abundance (for radioisotopes the half-life is given instead)
    3         fm      bound coherent scattering length
    4         fm      bound incoherent scattering length
    5         barn    bound coherent scattering cross section
    6         barn    bound incoherent scattering cross section
    7         barn    total bound scattering cross section
    8         barn    absorption cross section for 2200 m/s neutrons

    Note: 1 fm = 1E-15 m, 1 barn = 1E-24 cm^2, scattering legnths and cross sections in parenthesis are uncertainties
    '''

    from pkg_resources import resource_stream

    if rawTable:
        stream = resource_stream(__name__,'Data/NeutronScatteringLengths.csv')
        nsl_DF = pd.read_csv(stream)
        return nsl_DF


    stream = resource_stream(__name__,'Data/NeutronScatteringLengths_Corrected.csv')
    nsl_DF = pd.read_csv(stream).astype({'Isotope':'string',
                                        'Conc':np.float64,
                                        'Coh b':np.complex64,
                                        'Inc b':np.complex64,
                                        'Coh xs':np.float64,
                                        'Inc xs':np.float64,
                                        'Scatt xs':np.float64,
                                        'Abs xs':np.float64})
    return nsl_DF
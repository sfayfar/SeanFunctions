"""
scattering.py
----
Collection of tools useful to neutron and X-ray scattering.
"""

import re

import mendeleev as pTable
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import constants
from scipy.interpolate import interp1d

from .math import find_nearest, fourierbesseltransform


def atomic_form_factor_constants():
    """

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

    """
    from pkg_resources import resource_stream

    stream = resource_stream(__name__, "Data/AtomicFormFactorConstants.csv")
    aff_DF = pd.read_csv(stream)
    return aff_DF


def atomic_form_factor(atom, QList, inputCoeff=None):
    """
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
    """

    aff_DF = atomic_form_factor_constants()

    if atom not in aff_DF["element"].values:
        print(
            f"The atom '{atom}' is not available.\nThe available atoms are:\n",
            "\t".join(map(str, aff_DF["element"].values)),
        )
    else:
        if inputCoeff is None:
            coeff_values = aff_DF[aff_DF["element"] == atom].iloc[:, 2:-2].values[0]
        else:
            coeff_values = inputCoeff
        sumData = 0.0
        for i in range(0, len(coeff_values) - 1, 2):
            sumData += coeff_values[i] * np.exp(
                -coeff_values[i + 1] * (QList / (4 * np.pi)) ** 2
            )
        sumData += coeff_values[-1]
        return sumData


def neutron_scattering_lengths(rawTable=False):
    """
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

    Note: 1 fm = 1E-15 m, 1 barn = 1E-24 cm^2, scattering legnths and cross sections in parenthesis are uncertainties.
    """

    from pkg_resources import resource_stream

    if rawTable:
        stream = resource_stream(__name__, "Data/NeutronScatteringLengths.csv")
        nsl_DF = pd.read_csv(stream)
        return nsl_DF

    stream = resource_stream(__name__, "Data/NeutronScatteringLengths_Corrected.csv")
    nsl_DF = pd.read_csv(stream).astype(
        {
            "Isotope": "string",
            "Conc": np.float64,
            "Coh b": np.complex64,
            "Inc b": np.complex64,
            "Coh xs": np.float64,
            "Inc xs": np.float64,
            "Scatt xs": np.float64,
            "Abs xs": np.float64,
        }
    )
    return nsl_DF


class weight_RDF_for_scattering:
    def __init__(
        self,
        RDF_DataFrame,
        composition,
        cutoffR=None,
        isotopeDict=None,
        ionsDict=None,
        interpType="cubic",
        interpAmount=10,
        xrayRCut=100,
    ):
        """
        Converts molecular dynamics partial RDFs into weighted g(r) and S(Q) for neutron and X-ray scattering.
        The input RDF must be a Pandas DataFrame with the column names as the atomic pairs such as "F-F" or "Na-Cl".

        Parameters
        ---------
        RDF_DataFrame : DataFrame
                    Pandas DataFrame with the partial RDFs

        composition : str
                    Input the composition as a string.
                    Spaces are not required.
                    Ex: F4Li2Be

        cutoffR : float, optional
                Optionally cutoff the RDF at a value of r for calculations.

        isotopeDict : dict, optional
                    Add the isotopes of each atom as a Python dictionary.
                    The dict must start with the atom then include isotopes
                    The total must sum to 1.
                    Ex: {'Li':{'7Li':0.9,'6Li':0.1}}

        ionsDict : dict, optional
                Add the ionic charge for the form factors rather than
                the atomic form factors.
                Ex: {'Li':'1+','Be':'2+','F':'1-'}

        interpType : string, optional
                The type of interpolation to use with Scipy's interp1d function.
                Options: 'linear', 'quadratic', 'cubic' (default)

        interpAmount : int, optional
                The multiple amount of data points to add.
                Default value of 10.

        Returns
        --------
        compositionTable : DataFrame
                        DateFrame containing the composition, b, and f(Q) values.

        partialRDF : DataFrame

        unweightedSofQ : DataFrame
        """
        self.isotopeDict = isotopeDict
        self.ionsDict = ionsDict
        neutronScatteringLengths = neutron_scattering_lengths()

        # First convert the inputed composition to an array with the atoms and their fractional composition
        comps = re.sub(r"([A-Z])", r" \1", composition.replace(" ", "")).split()
        compArr = []
        for atoms in comps:
            atomArr = re.split(r"([0-9.]+)", atoms)[:2]
            if len(atomArr) == 1:
                atomArr.append(1)
            atomArr[-1] = float(atomArr[-1])
            compArr.append(atomArr)
        compArr = np.array(compArr, dtype="object")
        atomArr = compArr[:, 0]
        concArr = compArr[:, 1]

        if cutoffR is not None:
            cutoff = find_nearest(RDF_DataFrame.iloc[:, 0], cutoffR)[0]
            RDF_DataFrame = RDF_DataFrame.iloc[:cutoff, :]
        self.partialRDF = RDF_DataFrame.rename(columns={RDF_DataFrame.keys()[0]: "r"})

        if self.isotopeDict is not None:
            bArr = []
            for atoms in atomArr:
                if atoms in self.isotopeDict.keys():
                    bValue = 0
                    for isotope in self.isotopeDict[atoms].keys():
                        bValue += (
                            self.isotopeDict[atoms][isotope]
                            * neutronScatteringLengths.loc[
                                neutronScatteringLengths["Isotope"].str.fullmatch(
                                    isotope
                                )
                            ]["Coh b"]
                            .values[0]
                            .real
                        )
                    bArr.append(bValue)
                else:
                    bArr.append(
                        neutronScatteringLengths.loc[
                            neutronScatteringLengths["Isotope"].str.fullmatch(atoms)
                        ]["Coh b"]
                        .values[0]
                        .real
                    )
        else:
            bArr = [
                neutronScatteringLengths.loc[
                    neutronScatteringLengths["Isotope"].str.fullmatch(atomArr[i])
                ]["Coh b"]
                .values[0]
                .real
                for i in range(len(atomArr))
            ]

        QArr, SofQ = fourierbesseltransform(
            RDF_DataFrame.iloc[:, 0], RDF_DataFrame.iloc[:, 1] - 1, unpack=True
        )
        self.QArrInterp = np.linspace(QArr[0], QArr[-1], len(QArr) * interpAmount)

        if self.ionsDict is not None:
            affArr = [
                atomic_form_factor(
                    atomArr[i] + self.ionsDict[atomArr[i]], self.QArrInterp
                )
                for i in range(len(atomArr))
            ]
        else:
            affArr = [
                atomic_form_factor(atomArr[i], self.QArrInterp)
                for i in range(len(atomArr))
            ]

        self.compositionTable = pd.DataFrame(
            {
                "conc": concArr / np.sum(concArr),
                "amu": [
                    pTable.element(atomArr[i]).atomic_weight
                    for i in range(len(atomArr))
                ],
                "b": bArr,
                "aff": affArr,
            },
            index=atomArr,
        )

        # First Fourier transform the partial RDFs to partial SofQs
        self.unweightedSofQ = pd.DataFrame()
        for column in RDF_DataFrame.keys()[1:]:
            Q, SofQ = fourierbesseltransform(
                RDF_DataFrame.iloc[:, 0], RDF_DataFrame[column] - 1, unpack=True
            )
            SofQ_interp = interp1d(
                Q, SofQ, kind=interpType, bounds_error=False, fill_value=np.nan
            )(self.QArrInterp)
            self.unweightedSofQ["Q"] = self.QArrInterp
            self.unweightedSofQ[column] = SofQ_interp

        def _weighting(self, df, type="b"):
            weightArray = {}
            weightTotal = 0
            for column in df.keys()[1:]:
                atoms = column.split("-")
                weighting = np.prod(
                    self.compositionTable.loc[atoms].conc.values
                    * self.compositionTable.loc[atoms][type].values
                ) * (2 - 1 * (atoms[0] == atoms[1]))
                weightArray[column] = weighting
                weightTotal += weighting
            return weightArray, weightTotal

        # Neutron weighting
        self.weightArrayNeutron, self.weightTotalNeutron = _weighting(
            self, RDF_DataFrame, type="b"
        )

        # Neutron gofr
        self.gofrNeutron = pd.DataFrame()
        self.gofrNeutron["r"] = RDF_DataFrame.iloc[:, 0]
        totalgofr = 0
        for column in RDF_DataFrame.keys()[1:]:
            self.gofrNeutron[column] = (
                self.weightArrayNeutron[column]
                * RDF_DataFrame[column]
                / self.weightTotalNeutron
            )
            totalgofr += self.gofrNeutron[column]
        self.gofrNeutron["Total"] = totalgofr

        # Neutron SofQ
        self.SofQNeutron = pd.DataFrame()
        self.SofQNeutron["Q"] = self.QArrInterp
        totalSofQ = 0
        for column in RDF_DataFrame.keys()[1:]:
            self.SofQNeutron[column] = (
                self.weightArrayNeutron[column]
                * self.unweightedSofQ[column]
                / self.weightTotalNeutron
            )
            totalSofQ += self.SofQNeutron[column]
        self.SofQNeutron["Total"] = totalSofQ

        # X-ray weighting
        self.weightArrayXray, self.weightTotalXray = _weighting(
            self, RDF_DataFrame, type="aff"
        )

        # X-ray SofQ
        self.SofQXray = pd.DataFrame()
        self.SofQXray["Q"] = self.QArrInterp
        totalSofQ = 0
        for column in RDF_DataFrame.keys()[1:]:
            self.SofQXray[column] = (
                self.weightArrayXray[column]
                * self.unweightedSofQ[column]
                / self.weightTotalXray
            )
            totalSofQ += self.SofQXray[column]
        self.SofQXray["Total"] = totalSofQ

        # X-ray gofr
        self.gofrXray = pd.DataFrame()
        # self.gofrXray['r'] = RDF_DataFrame.iloc[:,0]
        totalgofr = 0
        cutAt100InvAng = find_nearest(self.SofQXray["Q"], xrayRCut)[0]
        for column in RDF_DataFrame.keys()[1:]:
            r, gofr = fourierbesseltransform(
                self.SofQXray["Q"].iloc[:cutAt100InvAng:interpAmount].to_numpy(),
                self.SofQXray[column].iloc[:cutAt100InvAng:interpAmount].to_numpy(),
                unpack=True,
            )
            self.gofrXray["r"] = r
            self.gofrXray[column] = gofr * 2 / np.pi
            totalgofr += gofr * 2 / np.pi
        self.gofrXray["Total"] = totalgofr

    def plot_SofQNeutron(self, axes=None, **kwargs):
        ax = plt.axes(axes)
        ax.plot(
            self.SofQNeutron["Q"],
            self.SofQNeutron["Total"],
            "k-",
            lw=2,
            label="Total",
            **kwargs,
        )
        for column in self.SofQNeutron.keys()[1:-1]:
            ax.plot(
                self.SofQNeutron["Q"],
                self.SofQNeutron[column],
                "--",
                lw=1,
                label=column,
                **kwargs,
            )
        return ax

    def plot_gofrNeutron(self, axes=None, **kwargs):
        ax = plt.axes(axes)
        ax.plot(
            self.gofrNeutron["r"],
            self.gofrNeutron["Total"],
            "k-",
            lw=2,
            label="Total",
            **kwargs,
        )
        for column in self.gofrNeutron.keys()[1:-1]:
            ax.plot(
                self.gofrNeutron["r"],
                self.gofrNeutron[column],
                "--",
                lw=1,
                label=column,
                **kwargs,
            )
        return ax

    def plot_SofQXray(self, axes=None, **kwargs):
        ax = plt.axes(axes)
        ax.plot(
            self.SofQXray["Q"],
            self.SofQXray["Total"],
            "k-",
            lw=2,
            label="Total",
            **kwargs,
        )
        for column in self.SofQXray.keys()[1:-1]:
            ax.plot(
                self.SofQXray["Q"],
                self.SofQXray[column],
                "--",
                lw=1,
                label=column,
                **kwargs,
            )
        return ax

    def plot_gofrXray(self, axes=None, **kwargs):
        ax = plt.axes(axes)
        ax.plot(
            self.gofrXray["r"],
            self.gofrXray["Total"],
            "k-",
            lw=2,
            label="Total",
            **kwargs,
        )
        for column in self.gofrXray.keys()[1:-1]:
            ax.plot(
                self.gofrXray["r"],
                self.gofrXray[column],
                "--",
                lw=1,
                label=column,
                **kwargs,
            )
        return ax

    def plot_scatteringLengths(self, axes=None, xray=True, neutron=True, **kwargs):
        ax = plt.axes(axes)

        for atoms in self.compositionTable.index:
            if neutron:
                labelValue = atoms
                if self.isotopeDict is not None:
                    if atoms in self.isotopeDict.keys():
                        labelValue + str(self.isotopeDict[atoms])
                ax.plot(
                    self.QArrInterp,
                    np.full(len(self.QArrInterp), self.compositionTable.b[atoms]),
                    "-",
                    lw=1,
                    label=labelValue + " b",
                    **kwargs,
                )
            if xray:
                labelValue = atoms
                if self.ionsDict is not None:
                    if atoms in self.ionsDict.keys():
                        labelValue += str(self.ionsDict[atoms])
                ax.plot(
                    self.QArrInterp,
                    self.compositionTable.aff[atoms],
                    "-",
                    lw=1,
                    label=labelValue + " aff",
                    **kwargs,
                )

    def calc_num_density(self, density):
        """
        Calculates the number density (in num per Ang^3) using the
        chemical formula and input density.

        Inputs
        ------
        density : float
                The density of the material at the desired temperature.

        Returns
        -------
        num_density : float
        """
        num_density = (
            constants.Avogadro
            / 10**24
            * density
            / np.sum([self.compositionTable.conc * self.compositionTable.amu])
        )
        return num_density

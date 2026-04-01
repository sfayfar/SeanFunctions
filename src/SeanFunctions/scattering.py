"""
scattering.py
----
Collection of tools useful to neutron and X-ray scattering.
"""

import re

import numpy as np
import pandas as pd
import periodictable as pTable
from periodictable.cromermann import fxrayatq
from matplotlib import pyplot as plt
from scipy import constants
from scipy.interpolate import interp1d

from .math import find_nearest, fourierbesseltransform


ION_RE = re.compile(r"^([A-Z][a-z]?)(?:(\d*)([+-]))?$")
ISOTOPE_RE = re.compile(r"^(?:(\d+)([A-Z][a-z]?)|([A-Z][a-z]?)(\d+))$")


def _parse_ion_symbol(atom):
    match = ION_RE.fullmatch(atom)
    if match is None:
        raise ValueError(
            f"Invalid atom or ion '{atom}'. Expected forms like 'O', 'Ca2+', or 'H1-'."
        )

    symbol, magnitude, sign = match.groups()
    charge = None
    if sign is not None:
        charge = int(magnitude) if magnitude else 1
        if sign == "-":
            charge *= -1

    return symbol, charge


def _get_element(symbol):
    try:
        return pTable.elements.symbol(symbol)
    except ValueError as exc:
        raise ValueError(f"Unsupported element symbol '{symbol}'.") from exc


def _get_neutron_scattering_length(symbol):
    element = _get_element(symbol)
    neutron = element.neutron
    if neutron is None or neutron.b_c is None:
        raise ValueError(
            f"No neutron coherent scattering length is available for '{symbol}'."
        )
    return np.real(neutron.b_c)


def _parse_isotope_label(isotope):
    match = ISOTOPE_RE.fullmatch(isotope)
    if match is None:
        raise ValueError(
            f"Invalid isotope '{isotope}'. Expected forms like '7Li' or 'Li7'."
        )

    leading_mass, leading_symbol, trailing_symbol, trailing_mass = match.groups()
    if leading_symbol is not None:
        return leading_symbol, int(leading_mass)
    return trailing_symbol, int(trailing_mass)


def _get_isotope_scattering_length(atom, isotope):
    isotope_symbol, mass_number = _parse_isotope_label(isotope)
    if isotope_symbol != atom:
        raise ValueError(
            f"Isotope '{isotope}' does not match atom '{atom}' in isotopeDict."
        )

    element = _get_element(atom)
    neutron = element[mass_number].neutron
    if neutron is None or neutron.b_c is None:
        raise ValueError(
            f"No neutron coherent scattering length is available for isotope '{isotope}'."
        )
    return np.real(neutron.b_c)


def _parse_composition(composition, combine_duplicates=False):
    comps = re.sub(r"([A-Z])", r" \1", composition.replace(" ", "")).split()

    if combine_duplicates:
        comp_dict = {}
        for atoms in comps:
            atom_arr = re.split(r"([0-9.]+)", atoms)[:2]
            if len(atom_arr) == 1:
                atom_arr.append(1)
            comp_dict[atom_arr[0]] = comp_dict.get(atom_arr[0], 0.0) + float(
                atom_arr[-1]
            )
        atom_arr = np.array(list(comp_dict.keys()), dtype="object")
        conc_arr = np.array(list(comp_dict.values()), dtype=float)
        return atom_arr, conc_arr

    comp_arr = []
    for atoms in comps:
        atom_arr = re.split(r"([0-9.]+)", atoms)[:2]
        if len(atom_arr) == 1:
            atom_arr.append(1)
        atom_arr[-1] = float(atom_arr[-1])
        comp_arr.append(atom_arr)
    comp_arr = np.array(comp_arr, dtype="object")
    return comp_arr[:, 0], comp_arr[:, 1].astype(float)


def _build_neutron_scattering_array(atom_arr, isotope_dict=None):
    if isotope_dict is None:
        return [_get_neutron_scattering_length(atom) for atom in atom_arr]

    b_arr = []
    for atom in atom_arr:
        if atom in isotope_dict:
            b_value = 0.0
            for isotope, fraction in isotope_dict[atom].items():
                b_value += fraction * _get_isotope_scattering_length(atom, isotope)
            b_arr.append(b_value)
        else:
            b_arr.append(_get_neutron_scattering_length(atom))
    return b_arr


def _build_xray_scattering_array(atom_arr, q_arr, ions_dict=None):
    aff_arr = []
    for atom in atom_arr:
        if ions_dict is not None and atom in ions_dict:
            aff_arr.append(atomic_form_factor(atom + ions_dict[atom], q_arr))
        else:
            aff_arr.append(atomic_form_factor(atom, q_arr))
    return aff_arr


def _build_composition_table(atom_arr, conc_arr, q_arr, isotope_dict=None, ions_dict=None):
    return pd.DataFrame(
        {
            "conc": conc_arr / np.sum(conc_arr),
            "amu": [_get_element(atom).mass for atom in atom_arr],
            "b": _build_neutron_scattering_array(atom_arr, isotope_dict=isotope_dict),
            "aff": _build_xray_scattering_array(atom_arr, q_arr, ions_dict=ions_dict),
        },
        index=atom_arr,
    )


def _build_atomic_pairs(atoms):
    atomic_pairs = []
    for i in range(len(atoms)):
        for j in range(i, len(atoms)):
            atomic_pairs.append(f"{atoms[i]}-{atoms[j]}")
    return atomic_pairs


def _calc_weighting(composition_table, pair_names, type="b"):
    weight_array = {}
    weight_total = 0
    for pair_name in pair_names:
        atoms = pair_name.split("-")
        weighting = np.prod(
            composition_table.loc[atoms].conc.values
            * composition_table.loc[atoms][type].values
        ) * (2 - 1 * (atoms[0] == atoms[1]))
        weight_array[pair_name] = weighting
        weight_total += weighting
    return weight_array, weight_total


def atomic_form_factor(atom, QList):
    """
    Returns the Q dependent atomic form factor for each of the elements and ions.
    The values are calculated using periodictable's Cromer-Mann coefficients.

    Parameters
    -------
    atom : str
        Put the name of the atom (or ion) for the atomic form factor.
        Ions have their charge put directly after the atom in the form
        X#+/-
        For example: Pt4+ or H1-

    QList : array_like
        The array of Q values that the atomic form factor will be calculated for.
    """
    symbol, charge = _parse_ion_symbol(atom)
    _get_element(symbol)
    try:
        return fxrayatq(symbol, QList, charge=charge)
    except Exception as exc:
        raise ValueError(
            f"Unable to calculate the atomic form factor for '{atom}'."
        ) from exc


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

        atomArr, concArr = _parse_composition(composition, combine_duplicates=True)

        if cutoffR is not None:
            cutoff = find_nearest(RDF_DataFrame.iloc[:, 0], cutoffR)[0]
            RDF_DataFrame = RDF_DataFrame.iloc[:cutoff, :]
        self.partialRDF = RDF_DataFrame.rename(columns={RDF_DataFrame.keys()[0]: "r"})

        QArr, SofQ = fourierbesseltransform(
            RDF_DataFrame.iloc[:, 0], RDF_DataFrame.iloc[:, 1] - 1, unpack=True
        )
        self.QArrInterp = np.linspace(QArr[0], QArr[-1], len(QArr) * interpAmount)

        self.compositionTable = _build_composition_table(
            atomArr,
            concArr,
            self.QArrInterp,
            isotope_dict=self.isotopeDict,
            ions_dict=self.ionsDict,
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

        # Neutron weighting
        self.weightArrayNeutron, self.weightTotalNeutron = _calc_weighting(
            self.compositionTable, RDF_DataFrame.keys()[1:], type="b"
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
        self.weightArrayXray, self.weightTotalXray = _calc_weighting(
            self.compositionTable, RDF_DataFrame.keys()[1:], type="aff"
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
        totalgofr = 0

        # Sample the Q and S(Q) arrays with the given interpAmount
        Q_sampled = self.SofQXray["Q"].iloc[::interpAmount].to_numpy()
        # find nearest index in the sampled Q array to the requested cut
        cut_sample_idx = find_nearest(Q_sampled, xrayRCut)[0]

        for column in RDF_DataFrame.keys()[1:]:
            S_sampled = self.SofQXray[column].iloc[::interpAmount].to_numpy()
            # apply the sinc window (sin(pi*x)/(pi*x)) safely using numpy.sinc
            window = np.sinc(Q_sampled / xrayRCut)
            S_windowed = S_sampled * window

            # zero everything at and beyond the cut index instead of truncating
            if cut_sample_idx < len(S_windowed):
                S_windowed[cut_sample_idx:] = 0

            r, gofr = fourierbesseltransform(Q_sampled, S_windowed, unpack=True)

            self.gofrXray["r"] = r
            self.gofrXray[column] = gofr * 2 / np.pi
            totalgofr += gofr * 2 / np.pi

        self.gofrXray["Total"] = totalgofr

    def calc_NOMAD_prefactor(self, num_density:float=None, density:float=None):
        """
        Calculates the prefactor for the NOMAD format for S(Q). The prefactor is 4 * pi * sum_i (c_i * b_i)**2 / sum_i (c_i * b_i**2) where c_i is the concentration and b_i is the neutron scattering length of each atom.

        Parameters
        ---------
        num_density : float, optional
                    The number density of the material in num per Ang^3.
                    If not inputed then it will be calculated using the composition and density.

        density : float, optional
                The density of the material at the desired temperature in g/cm^3.
                Required if num_density is not inputed.

        Returns
        -------
        prefactor : float
                    The prefactor to convert S(Q) and g(r) to the NOMAD format.
        """
        if num_density is None:
            if density is None:
                raise ValueError(
                    "Either num_density or density must be inputed to calculate the prefactor."
                )
            else:
                num_density = self.calc_num_density(density)

        sum_c_b_total_squared = np.sum(
            self.compositionTable.conc * self.compositionTable.b
        ) ** 2
        sum_c_b_squared = np.sum(
            self.compositionTable.conc * self.compositionTable.b**2
        )

        prefactor = 4 * np.pi * num_density * sum_c_b_total_squared / sum_c_b_squared
        return prefactor

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
                        labelValue += str(self.isotopeDict[atoms])
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
        return ax

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


class ScatteringComposition:
    def __init__(
        self,
        composition,
        QArr=None,
        isotopeDict=None,
        ionsDict=None,
    ):
        """
        Builds composition-based neutron and X-ray scattering quantities
        without requiring RDF input.

        This class parses a chemical formula, computes the normalized
        composition, atomic masses, neutron coherent scattering lengths,
        X-ray atomic form factors, unique atomic pairs, and the neutron
        and X-ray weighting factors used by weight_RDF_for_scattering.
        It is useful when you want access to the scattering tables and
        pair-weight prefactors without calculating g(r) or S(Q).

        Parameters
        ----------
        composition : str
                    Input the composition as a string.
                    Spaces are not required.
                    Ex: F4Li2Be

        QArr : array_like, optional
                    Q grid to use for the X-ray atomic form factors.
                    Default: np.linspace(0, 20, 101)

        Returns
        -------
        compositionTable : DataFrame
                        DataFrame containing the composition, b, and f(Q) values.

        atomic_pairs : list
                        Unique atomic pair labels generated from the composition.

        weightArrayNeutron : dict
                        Neutron scattering weights for each atomic pair.

        weightArrayXray : dict
                        X-ray scattering weights for each atomic pair.
        """
        self.isotopeDict = isotopeDict
        self.ionsDict = ionsDict
        self.QArr = np.linspace(0, 20, 101) if QArr is None else np.asarray(QArr)

        atomArr, concArr = _parse_composition(composition, combine_duplicates=True)
        self.compositionTable = _build_composition_table(
            atomArr,
            concArr,
            self.QArr,
            isotope_dict=self.isotopeDict,
            ions_dict=self.ionsDict,
        )

        self.atomic_pairs = _build_atomic_pairs(self.compositionTable.index)
        self.weightArrayNeutron, self.weightTotalNeutron = _calc_weighting(
            self.compositionTable, self.atomic_pairs, type="b"
        )
        self.weightArrayXray, self.weightTotalXray = _calc_weighting(
            self.compositionTable, self.atomic_pairs, type="aff"
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


scatteringComposition = ScatteringComposition

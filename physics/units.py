# Copyright Anatole Hernot (Mines Paris), 2022. All rights reserved.

import numpy as np

# TODO: SORTING FUNCTION for self.__units
# TODO: __call__ function (make an unit_list parsing function to avoid copy-pasting the same code all over) => work with UNITS or DIMENSIONS?


UNITS = {"mass": {"": ""}}


UNITS_DICT = {
    'fg': 'mass',
    'pg': 'mass',
    'ng': 'mass',
    'mig': 'mass',
    'mg': 'mass',
    'cg':  'mass',
    'dg':  'mass',
    'g':   'mass',
    'dag': 'mass',
    'hg':  'mass',
    'kg':  'mass',
    't':   'mass'
}


# {'fg': 1e-18, 'pg': 1e-15, 'ng': 1e-12, 'mig': 1e-9, 'mg': 1e-6, 'cg': 1e-5, 'dg': 1e-4, 'g': 1e-3, 'dag': 1e-2, 'hg': 1e-1, 'kg': 1, 't': 1e3}

_U_MASS = {
    "fg": {
        "fg": "1e0",
        "pg": "1e3",
        "ng": "1e6",
        "mig": "1e9",
        "mg": "1e12",
        "cg": "1e13",
        "dg": "1e14",
        "g": "1e15",
        "dag": "1e16",
        "hg": "1e17",
        "kg": "1e18",
        "t": "1e21",
    },
    "pg": {
        "fg": "1e-3",
        "pg": "1e0",
        "ng": "1e3",
        "mig": "1e6",
        "mg": "1e9",
        "cg": "1e10",
        "dg": "1e11",
        "g": "1e12",
        "dag": "1e13",
        "hg": "1e14",
        "kg": "1e15",
        "t": "1e18",
    },
    "ng": {
        "fg": "1e-6",
        "pg": "1e-3",
        "ng": "1e0",
        "mig": "1e3",
        "mg": "1e6",
        "cg": "1e7",
        "dg": "1e8",
        "g": "1e9",
        "dag": "1e10",
        "hg": "1e11",
        "kg": "1e12",
        "t": "1e15",
    },
    "mig": {
        "fg": "1e-9",
        "pg": "1e-6",
        "ng": "1e-3",
        "mig": "1e0",
        "mg": "1e3",
        "cg": "1e4",
        "dg": "1e5",
        "g": "1e6",
        "dag": "1e7",
        "hg": "1e8",
        "kg": "1e9",
        "t": "1e12",
    },
    "mg": {
        "fg": "1e-12",
        "pg": "1e-9",
        "ng": "1e-6",
        "mig": "1e-3",
        "mg": "1e0",
        "cg": "1e1",
        "dg": "1e2",
        "g": "1e3",
        "dag": "1e4",
        "hg": "1e5",
        "kg": "1e6",
        "t": "1e9",
    },
    "cg": {
        "fg": "1e-13",
        "pg": "1e-10",
        "ng": "1e-7",
        "mig": "1e-4",
        "mg": "1e-1",
        "cg": "1e0",
        "dg": "1e1",
        "g": "1e2",
        "dag": "1e3",
        "hg": "1e4",
        "kg": "1e5",
        "t": "1e8",
    },
    "dg": {
        "fg": "1e-14",
        "pg": "1e-11",
        "ng": "1e-8",
        "mig": "1e-5",
        "mg": "1e-2",
        "cg": "1e-1",
        "dg": "1e0",
        "g": "1e1",
        "dag": "1e2",
        "hg": "1e3",
        "kg": "1e4",
        "t": "1e7",
    },
    "g": {
        "fg": "1e-15",
        "pg": "1e-12",
        "ng": "1e-9",
        "mig": "1e-6",
        "mg": "1e-3",
        "cg": "1e-2",
        "dg": "1e-1",
        "g": "1e0",
        "dag": "1e1",
        "hg": "1e2",
        "kg": "1e3",
        "t": "1e6",
    },
    "dag": {
        "fg": "1e-16",
        "pg": "1e-13",
        "ng": "1e-10",
        "mig": "1e-7",
        "mg": "1e-4",
        "cg": "1e-3",
        "dg": "1e-2",
        "g": "1e-1",
        "dag": "1e0",
        "hg": "1e1",
        "kg": "1e2",
        "t": "1e5",
    },
    "hg": {
        "fg": "1e-17",
        "pg": "1e-14",
        "ng": "1e-11",
        "mig": "1e-8",
        "mg": "1e-5",
        "cg": "1e-4",
        "dg": "1e-3",
        "g": "1e-2",
        "dag": "1e-1",
        "hg": "1e0",
        "kg": "1e1",
        "t": "1e4",
    },
    "kg": {
        "fg": "1e-18",
        "pg": "1e-15",
        "ng": "1e-12",
        "mig": "1e-9",
        "mg": "1e-6",
        "cg": "1e-5",
        "dg": "1e-4",
        "g": "1e-3",
        "dag": "1e-2",
        "hg": "1e-1",
        "kg": "1e0",
        "t": "1e3",
    },
    "t": {
        "fg": "1e-21",
        "pg": "1e-18",
        "ng": "1e-15",
        "mig": "1e-12",
        "mg": "1e-9",
        "cg": "1e-8",
        "dg": "1e-7",
        "g": "1e-6",
        "dag": "1e-5",
        "hg": "1e-4",
        "kg": "1e-3",
        "t": "1e0",
    },
}


# generate array (decimal system)
def gen():
    units = ["fg", "pg", "ng", "mig", "mg", "cg", "dg", "g", "dag", "hg", "kg", "t"]
    mass_dec = np.array([-18, -15, -12, -9, -6, -5, -4, -3, -2, -1, 0, 3])
    masses_array = np.empty((mass_dec.shape[0], mass_dec.shape[0]))

    _u_mass = dict()
    for i, unit in enumerate(units):
        m_offset = mass_dec[i]
        masses = mass_dec - m_offset
        masses_array[i] = masses

        # Manually fill line
        _u_mass[unit] = dict()
        for j, unit_2 in enumerate(units):
            _u_mass[unit][unit_2] = f"1e{-1 * masses[j]}"

        # _u_mass [unit] = dict(zip(units, np.power(10, masses)))

    print(masses_array)
    print(_u_mass)


# TODO: make a script that works with powers of 10 only, with exact results (combine into float at the end)
# TODO: make an addition to deal with non-base10 systems
class DimensionedValue_old:
    def __init__(self, val, unit: str):
        self.__val = val
        self.__unit = unit.lower()

        # TODO: find unit in units and thus find conversion dict
        self.__conversion_dict = _U_MASS
        

    def __repr__(self):
        return f"{self.__val} {self.__unit}"

    def __call__(self, unit: str) -> float or int:
        multiplier = float(self.__conversion_dict [self.__unit] [unit])
        return self.__val * multiplier





#################





import json

with open('physics/units.json', 'r', encoding='utf-8') as units_db:
    UNIT_CONVERSIONS = json.load(units_db)



# DimensionedValue (10, kg)
# DimensionedValue (10, (kg, 1))
# DimensionedValue (10, (kg, 1), (m, 2))
# ! DimensionedValue (10, (kg, m), (1, 2))
# DimensionedValue (10, kg^2)  # string parsing
# WHAT IF DV(10kg, 7g)???? wtf


def dimension_sort (dimension):
    pass

def parse_units_list (*units):
    pass

class DimensionedValue:
    """
    base-10 units only
    case-sensitive

    methods from https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types
    """

    
    def __init__ (self, val, *units: list or tuple):
        """ no string parsing for unit """
    
        self.__val = val
        self.__units_dict = dict()

        # Process units
        if isinstance(units[0], float) or isinstance(units[0], int):
            raise NotImplementedError  # unit-less number
        else:
            for unit_tuple in units:

                # Unpack unit
                if isinstance(unit_tuple, str):
                    unit, pwr = unit_tuple, 1
                elif isinstance(unit_tuple, tuple) or isinstance(unit_tuple, list):
                    unit, pwr = unit_tuple
                else:
                    raise ValueError

                # Add to dictionary
                if unit in self.__units_dict.keys():
                    self.__units_dict[unit] += pwr
                else:
                    self.__units_dict[unit] = pwr

        # Generate list of units
        self.__units = list(self.__units_dict.keys())
        self.__units.sort()

        # Generate unit str
        unit_strlist = list()
        for unit in self.__units:
            pwr = self.__units_dict[unit]
            if pwr == 0: continue
            unit_strlist.append(unit)
            if pwr != 1:
                unit_strlist[-1] += f'^{pwr}'
        self.__unit_str = '.'.join(unit_strlist)

        # Generate dimension
        self.dimension = list()
        for unit in self.__units:

            # Match known unit to its dimension
            try: dim = UNITS_DICT[unit]
            except KeyError: raise ValueError(f'unit "{unit}" not supported')
            
            pwr = self.__units_dict[unit]
            if pwr == 0: continue
            self.dimension .append((dim, pwr))

        print(f'\tDEBUG - {self.__units}')
        print(f'\tDEBUG - {self.dimension}')

        self.__conversion_dicts = dict()
        for dim, pwr in self.dimension:
            if pwr == 0: continue
            conv_dict = UNIT_CONVERSIONS[dim]
            conv_dict_scaled = dict((unit, mult**pwr) for unit, mult in conv_dict.items())
            self.__conversion_dicts[dim] = conv_dict_scaled

        

    def __repr__ (self):
        # return f"{self.__val} {self.__unit}"
        return f'{self.__val} {self.__unit_str}'

    # def __call__ (self, unit: str) -> float or int:
    #     multiplier = float(self.__conversion_dict [self.__unit] [unit])
    #     return self.__val * multiplier

    def __call__ (self, *units: list or tuple):

        pass



    def __neg__ (self):
        return DimensionedValue (self.__val.__neg__(), self.__unit)
    def __pos__ (self):
        return DimensionedValue (self.__val.__pos__(), self.__unit)
    def __abs__ (self):
        return DimensionedValue (self.__val.__abs__(), self.__unit)
    def __invert__ (self):
        return DimensionedValue (self.__val.__invert__(), self.__unit)

    def __round__ (self, ndigits: int):
        return DimensionedValue(self.__val.__round__(ndigits=ndigits), self.__unit)
    def __trunc__ (self):
        return DimensionedValue(self.__val.__trunc__(), self.__unit)
    def __floor__ (self):
        return DimensionedValue(self.__val.__floor__(), self.__unit)
    def __ceil__ (self):
        return DimensionedValue(self.__val.__ceil__(), self.__unit)



    def __add__ (self, other):
        if isinstance(other, DimensionedValue):
            if self.dimension != other.dimension: raise ValueError('Adding objects of different dimensions')
            return DimensionedValue(self.__val + other(self.__unit), self.__unit)
        elif isinstance(other, float) or isinstance(other, int):
            return DimensionedValue(self.__val + other, self.__unit)  # assuming that the units are the same
        else:
            return NotImplementedError
    def __radd__ (self, other):
        return self.__add__(other)
    def __sub__ (self, other): 
        return self .__add__(other.__neg__())
    def __rsub__ (self, other):
        return self.__neg__() .__radd__(other)
        

    def __mul__ (self, other):
        if isinstance(other, DimensionedValue):
            raise NotImplementedError  # to fix
        elif isinstance(other, float) or isinstance(other, int):
            return DimensionedValue(self.__val * other, self.__unit)
        else:
            return NotImplementedError
    def __rmul__ (self, other):
        return self.__mul__(other)

    def __truediv__ (self, other):
        if isinstance(other, DimensionedValue):
            raise NotImplementedError
            return DimensionedValue(self.__val__ .__truediv__(other.__val), self.__unit)
        elif isinstance(other, float) or isinstance(other, int):
            return DimensionedValue(self.__val .__truediv__(other), self.__unit)
        else:
            return NotImplementedError
    def __rtruediv__ (self, other):
        if isinstance(other, DimensionedValue):
            raise NotImplementedError
        elif isinstance(other, float) or isinstance(other, int):
            raise NotImplementedError
            return DimensionedValue(other.__truediv__(self.__val), self.__unit)
        else:
            return NotImplementedError


    def __lt__ (self, other):
        if isinstance(other, DimensionedValue):
            if self.dimension != other.dimension: raise ValueError('Comparing objects of different dimensions')
            return self.__val < other(self.__unit)
        elif isinstance(other, float) or isinstance(other, int):
            return self.__val < other
        else:
            return NotImplementedError
    def __le__ (self, other):
        if isinstance(other, DimensionedValue):
            if self.dimension != other.dimension: raise ValueError('Comparing objects of different dimensions')
            return self.__val <= other(self.__unit)
        elif isinstance(other, float) or isinstance(other, int):
            return self.__val <= other
        else:
            return NotImplementedError
    def __eq__ (self, other):
        if isinstance(other, DimensionedValue):
            if self.dimension != other.dimension: raise ValueError('Comparing objects of different dimensions')
            return self.__val == other(self.__unit)
        elif isinstance(other, float) or isinstance(other, int):
            return self.__val == other
        else:
            return NotImplementedError
    def __ne__ (self, other):
        if isinstance(other, DimensionedValue):
            if self.dimension != other.dimension: raise ValueError('Comparing objects of different dimensions')
            return self.__val != other(self.__unit)
        elif isinstance(other, float) or isinstance(other, int):
            return self.__val != other
        else:
            return NotImplementedError
    def __gt__ (self, other):
        if isinstance(other, DimensionedValue):
            if self.dimension != other.dimension: raise ValueError('Comparing objects of different dimensions')
            return self.__val > other(self.__unit)
        elif isinstance(other, float) or isinstance(other, int):
            return self.__val > other
        else:
            return NotImplementedError
    def __ge__ (self, other):
        if isinstance(other, DimensionedValue):
            if self.dimension != other.dimension: raise ValueError('Comparing objects of different dimensions')
            return self.__val >= other(self.__unit)
        elif isinstance(other, float) or isinstance(other, int):
            return self.__val >= other
        else:
            return NotImplementedError


    
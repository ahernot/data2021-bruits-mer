import numpy as np

UNITS = {"mass": {"": ""}}


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
            _u_mass[unit][unit_2] = f"1e{masses[j]}"

        # _u_mass [unit] = dict(zip(units, np.power(10, masses)))

    print(masses_array)
    print(_u_mass)



# TODO: make a script that works with powers of 10 only, with exact results (combine into float at the end)
# TODO: make an addition to deal with non-base10 systems
class DimensionedValue:
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



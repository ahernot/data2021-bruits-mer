
import numpy as np

UNITS = {
    'mass': {
        '': ''
    }
}



# {'fg': 1e-18, 'pg': 1e-15, 'ng': 1e-12, 'mig': 1e-9, 'mg': 1e-6, 'cg': 1e-5, 'dg': 1e-4, 'g': 1e-3, 'dag': 1e-2, 'hg': 1e-1, 'kg': 1, 't': 1e3}

_u_mass = {
    'fg':  {'fg': 1,     'pg': 1e3, 'ng': 1e6, 'mig': 1e-9, 'mg': 1e-6, 'cg': 1e-5, 'dg': 1e-4, 'g': 1e-3, 'dag': 1e-2, 'hg': 1e-1, 'kg': 1, 't': 1e3},
    'pg':  {'fg': 1e-3,  'pg': 1e-15, 'ng': 1e-12, 'mig': 1e-9, 'mg': 1e-6, 'cg': 1e-5, 'dg': 1e-4, 'g': 1e-3, 'dag': 1e-2, 'hg': 1e-1, 'kg': 1, 't': 1e3},
    'ng':  {'fg': 1e-6,  'pg': 1e-15, 'ng': 1e-12, 'mig': 1e-9, 'mg': 1e-6, 'cg': 1e-5, 'dg': 1e-4, 'g': 1e-3, 'dag': 1e-2, 'hg': 1e-1, 'kg': 1, 't': 1e3},
    'mig': {'fg': 1e-9,  'pg': 1e-15, 'ng': 1e-12, 'mig': 1e-9, 'mg': 1e-6, 'cg': 1e-5, 'dg': 1e-4, 'g': 1e-3, 'dag': 1e-2, 'hg': 1e-1, 'kg': 1, 't': 1e3},
    'mg':  {'fg': 1e-12, 'pg': 1e-15, 'ng': 1e-12, 'mig': 1e-9, 'mg': 1e-6, 'cg': 1e-5, 'dg': 1e-4, 'g': 1e-3, 'dag': 1e-2, 'hg': 1e-1, 'kg': 1, 't': 1e3},
    'cg':  {'fg': 1e-13, 'pg': 1e-15, 'ng': 1e-12, 'mig': 1e-9, 'mg': 1e-6, 'cg': 1e-5, 'dg': 1e-4, 'g': 1e-3, 'dag': 1e-2, 'hg': 1e-1, 'kg': 1, 't': 1e3},
    'dg':  {'fg': 1e-14, 'pg': 1e-15, 'ng': 1e-12, 'mig': 1e-9, 'mg': 1e-6, 'cg': 1e-5, 'dg': 1e-4, 'g': 1e-3, 'dag': 1e-2, 'hg': 1e-1, 'kg': 1, 't': 1e3},
    'g':   {'fg': 1e-15, 'pg': 1e-15, 'ng': 1e-12, 'mig': 1e-9, 'mg': 1e-6, 'cg': 1e-5, 'dg': 1e-4, 'g': 1e-3, 'dag': 1e-2, 'hg': 1e-1, 'kg': 1, 't': 1e3},
    'dag': {'fg': 1e-16, 'pg': 1e-15, 'ng': 1e-12, 'mig': 1e-9, 'mg': 1e-6, 'cg': 1e-5, 'dg': 1e-4, 'g': 1e-3, 'dag': 1e-2, 'hg': 1e-1, 'kg': 1, 't': 1e3},
    'hg':  {'fg': 1e-17, 'pg': 1e-15, 'ng': 1e-12, 'mig': 1e-9, 'mg': 1e-6, 'cg': 1e-5, 'dg': 1e-4, 'g': 1e-3, 'dag': 1e-2, 'hg': 1e-1, 'kg': 1, 't': 1e3},
    'kg':  {'fg': 1e-18, 'pg': 1e-15, 'ng': 1e-12, 'mig': 1e-9, 'mg': 1e-6, 'cg': 1e-5, 'dg': 1e-4, 'g': 1e-3, 'dag': 1e-2, 'hg': 1e-1, 'kg': 1, 't': 1e3},
    't':   {'fg': 1e-21, 'pg': 1e-15, 'ng': 1e-12, 'mig': 1e-9, 'mg': 1e-6, 'cg': 1e-5, 'dg': 1e-4, 'g': 1e-3, 'dag': 1e-2, 'hg': 1e-1, 'kg': 1, 't': 1e3},
}

# Generate array (decimal system)
mass_dec = {'fg': -18, 'pg': -15, 'ng': -12, 'mig': -9, 'mg': -6, 'cg': -5, 'dg': -4, 'g': -3, 'dag': -2, 'hg': -1, 'kg': 0, 't': 3}
units = ['fg', 'pg', 'ng', 'mig', 'mg', 'cg', 'dg', 'g', 'dag', 'hg', 'kg', 't']
mass_dec = np.array([-18, -15, -12, -9, -6, -5, -4, -3, -2, -1, 0, 3])
masses_array = np.empty((mass_dec.shape[0], mass_dec.shape[0]))

_u_mass = dict()
for i, unit in enumerate(units):
    m_offset = mass_dec[i]
    masses = mass_dec - m_offset
    masses_array[i] = masses

    # Manually fill line
    _u_mass [unit] = dict()
    for j, unit_2 in enumerate(units):
        _u_mass [unit] [unit_2] = f'1e{masses[j]}'

    # _u_mass [unit] = dict(zip(units, np.power(10, masses)))
    
print(masses_array)
print(_u_mass)



class DimensionedValue:

    def __init__ (self, val, unit: str):
        self.val = val
        self.unit = unit.lower()

    def __repr__ (self):
        return f'{self.val} {self.unit}'

    def __call__(self, unit: str) -> float or int:
        pass
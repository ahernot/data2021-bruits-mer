# Copyright Anatole Hernot (Mines Paris), 2022. All rights reserved._
import numpy as np
from scipy import interpolate
from scipy.optimize import fsolve
from scipy.misc import derivative

from absorption import calc_absorption
from physics.sound_propagation import sound_velocity_medwin
from profiles import *



class Propagation2D:

    __event_propagation = '_propagation_'
    __event_reflection  = '_reflection_'

    def __init__ (self, z0, theta_0, **kwargs):
        """
        :param z0: Source altitude
        :param theta_0: Emission angle

        **kwargs
        :param dz: Vertical resolution (in m)
        :param limit_bottom: …

        :param calc_c: Sound velocity calculator (return result in m.s^-1)
        :param n_steps_max: Maximum number of calculation steps

        :param reflection_coef_ground: default 0.6
        :param reflection_coef_water: default 1.0

        :param calc_der:
        :param func_solve:
        """

        # Set simulation parameters
        self.z0 = z0
        self.theta_0 = theta_0
        self.dz = kwargs.get('dz', 10)
        self.n_steps_max = kwargs.get('n_steps_max', 10000)
        self.reflection_coef_ground = kwargs.get('reflection_coef_ground', 0.6)
        self.reflection_coef_water = kwargs.get('reflection_coef_water', 1.0)
        
        # Set limits
        self.limit_top_val = 0.
        self.__init_limit_bottom(kwargs.get('limit_bottom', -1e4))

        # Set functions
        self.calc_der = kwargs.get('calc_der', derivative)
        self.func_solve = kwargs.get('func_solve', fsolve)

        # Run path simulation
        self.__generate()
        self.A = dict()
    

    def __init_limit_bottom (self, limit_bottom):
        if isinstance(limit_bottom, int) or isinstance(limit_bottom, float):
            def limit_bottom_func(x):
                if isinstance(x, np.ndarray): return np.ones(x.shape[0]) * limit_bottom
                else: return limit_bottom
            self.limit_bottom_func = limit_bottom_func
        elif isinstance(limit_bottom, np.ndarray):
            self.limit_bottom_func = interpolate.interp1d(limit_bottom[0], limit_bottom[1], kind='linear')  # linear interpolation
        else:
            raise ValueError('Wrong limit format')


    def __generate (self):

        # Initialise differential solver parameters
        c0 = calc_c(0)
        mult = -1 * np.power(c0 / np.sin(self.theta_0), 2)  # differential equation multiplier
        self.X = np.array([0, ])
        self.Z = np.array([self.z0, ])
        dx_z   = 1 / np.tan(self.theta_0)
        dxdx_z = 0  # no initial curvature

        i = 0
        self.events = [[Propagation2D.__event_propagation, np.empty((0, 2))]]
        while i < self.n_steps_max:

            # Calculate new point
            x = self.X[i]
            z = self.Z[i]
            dz = np.sign(dx_z) * self.dz
            dx = dz / dx_z
            x_new = x + dx
            z_new = z + dz
            
            # Check top reflection
            if z_new > self.limit_top_val:
                # Calculate intersection
                z_new = self.limit_top_val
                x_new = x + (self.limit_top_val - z) / dx_z
                dx_z *= -1
                # Add reflection event
                self.events.append([Propagation2D.__event_reflection, 1.0])
                self.events.append([Propagation2D.__event_propagation, np.empty((0, 2))])

            # Check bottom reflection
            elif z_new < self.limit_bottom_func(x_new):
                # Calculate intersection
                x_new = float(self.func_solve(lambda x1: self.limit_bottom_func(x) - dx_z * (x1 - x) - z, x0=x))
                z_new = self.limit_bottom_func(x_new)
                # Calculate reflection direction
                dx_z_ground = self.calc_der(self.limit_bottom_func, x_new, dx=x_new-x)  # dx_z_ground = (z_new - limit_bottom_func(z)) / (x_new - x)
                alpha_new = np.pi - np.arctan(dx_z) - 2 * np.arctan(dx_z_ground)
                dx_z = np.tan(alpha_new)
                # Add reflection event
                self.events.append([Propagation2D.__event_reflection, 0.6])
                self.events.append([Propagation2D.__event_propagation, np.empty((0, 2))])
                
            # Check backwards propagation
            if x_new < x: break  # prevent backwards reflection (to change)


            # Add new point
            self.X = np.concatenate((self.X, np.array([x_new, ])), axis=0)
            self.Z = np.concatenate((self.Z, np.array([z_new, ])), axis=0)
            # Calculate new path segment
            dx_new = x_new - x
            dz_new = z_new - z
            dl = np.sqrt(np.power(dx_new, 2) + np.power(dz_new, 2))
            self.events[-1][1] = np.concatenate((self.events[-1][1], np.array([[z, dl]])), axis=0)


            # Calculate new point's properties
            c = calc_c(z)
            g = calc_dz_c(z)
            # Update derivatives
            dxdx_z = mult * g / np.power(c, 3)
            dx_z  += dxdx_z * dx

            # Increment step counter
            i += 1

        # Generate interpolated path function
        self.Z_func = interpolate.interp1d(self.X, self.Z, kind='linear')


    def run (self, *freqs: float or int):

        for f in freqs:
            self.A[f] = np.array([0., ])

            reflection_coef_dB = 0.
            for event in self.events:

                if event[0] == Propagation2D.__event_propagation:
                    z, dl = event[1][:, 0], event[1][:, 1]
                    absorption_dB = np.multiply(calc_absorption(f, z, calc_T(z), calc_S(z), calc_pH(z)), dl)
                    absorption_dB_cum = reflection_coef_dB + self.A[f][-1] + np.cumsum(absorption_dB)
                    self.A[f] = np.concatenate((self.A[f], absorption_dB_cum), axis=0)
                    absorption_dB_cum = 0.

                elif event[0] == Propagation2D.__event_reflection:
                    reflection_coef_dB = 10 * np.log10(event[1])


    def gen_filter (self, x, fmin, fmax, nsamples):
        """
        :param x: Distance from source (in m)
        :param fmin: Min frequency
        :param fmax: Max frequency
        :param nsamples: Number of samples
        """

        f = np.linspace(fmin, fmax, nsamples)
        z = self.Z_func(x)

        absorption_dB = calc_absorption(f, z, calc_T(z), calc_S(z), calc_pH(z))



# model = propagation (environment parameters)
# => propagate sound over set distance (with dz=10m)
# => generate a sampling of (z, dl) to create attenuation filter from => call() gives a filter for a set propagation distance (on the x-axis)


"""
ground reflection coefficient!!!!!!!!!!!!!!

"""


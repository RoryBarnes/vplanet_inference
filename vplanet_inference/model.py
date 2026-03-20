import vplanet
import numpy as np 
import os
import re
import subprocess
import shutil
import time
import random
import astropy.units as u
from collections import OrderedDict

__all__ = ["VplanetModel"]

# VPLanet unit string -> astropy unit (per sUnit* key, case-insensitive)
_VPLANET_UNITS = {
    'sUnitMass': {
        'gram': u.g, 'g': u.g, 'kg': u.kg,
        'earth': u.M_earth, 'jupiter': u.M_jup,
        'solar': u.M_sun, 'sun': u.M_sun,
    },
    'sUnitLength': {
        'cm': u.cm, 'm': u.m, 'km': u.km,
        'earth': u.R_earth, 'jupiter': u.R_jup,
        'solar': u.R_sun, 'sun': u.R_sun, 'au': u.AU,
    },
    'sUnitTime': {
        'sec': u.s, 'second': u.s, 'seconds': u.s, 's': u.s,
        'day': u.d, 'days': u.d, 'd': u.d,
        'year': u.yr, 'years': u.yr, 'yr': u.yr,
        'myr': 1e6 * u.yr, 'gyr': 1e9 * u.yr,
    },
    'sUnitAngle': {
        'deg': u.deg, 'degrees': u.deg, 'd': u.deg, 'degree': u.deg,
        'rad': u.rad, 'radians': u.rad, 'radian': u.rad,
    },
    'sUnitTemp': {
        'k': u.K, 'kelvin': u.K,
    },
}

_SI_UNITS = {
    'sUnitMass': u.kg, 'sUnitLength': u.m, 'sUnitTime': u.s,
    'sUnitAngle': u.rad, 'sUnitTemp': u.K,
}

# Known VPLanet parameters -> which sUnit* key governs them
_PARAM_UNIT_KEYS = {
    'dObliquity': 'sUnitAngle', 'dPrecA': 'sUnitAngle',
    'dLongP': 'sUnitAngle', 'dLongA': 'sUnitAngle',
    'dInc': 'sUnitAngle', 'dArgP': 'sUnitAngle',
    'dMass': 'sUnitMass',
    'dSemi': 'sUnitLength', 'dRadius': 'sUnitLength',
    'dAge': 'sUnitTime', 'dTimeStep': 'sUnitTime',
}


def _get_unit_key(param_name):
    """Return the sUnit* key for a VPLanet parameter, or None."""
    if param_name in _PARAM_UNIT_KEYS:
        return _PARAM_UNIT_KEYS[param_name]
    lower = param_name.lower()
    if lower.endswith('time') or lower.endswith('period'):
        return 'sUnitTime'
    return None


class VplanetModel(object):

    def __init__(self, 
                 inparams, 
                 inpath=".", 
                 outparams=None, 
                 outpath="output/", 
                 fixsub=None,
                 executable="vplanet",
                 vplfile="vpl.in", 
                 sys_name="system", 
                 timesteps=None, 
                 time_init=5e6*u.yr, 
                 forward=True,
                 verbose=True):
        """
        Class for creating and executing VPLANET infiles.

        params  : (str, list) variable parameter names
                  ['vpl.dStopTime', 'star.dRotPeriod', 'star.dMass', 'planet.dEcc', 'planet.dOrbPeriod']
                
        inpath  : (str) path to template infiles
                  'infiles/'

        outparams : (str, list) return specified list of parameters from log file
                    ['final.primary.Radius', 'final.secondary.Radius']
        
        timesteps : (float * astropy units, optional)
        """

        # Input parameters
        self.inparams = list(inparams.keys())
        self.in_units = list(inparams.values())
        self.inpath = inpath
        self.executable = executable
        self.vplfile = vplfile
        self.sys_name = sys_name
        self.ninparam = len(inparams)
        self.verbose = verbose

        # Output parameters
        if outparams is not None:
            # format into alphabetized ordered dictionary 
            outparams = OrderedDict(sorted(outparams.items()))

            self.outparams = list(outparams.keys())
            self.out_units = list(outparams.values())
            self.noutparam = len(self.outparams)
        self.outpath_base = outpath

        # List of infiles (vpl.in + body files)
        self.infile_list = [vplfile]
        with open(os.path.join(inpath, vplfile), 'r') as vplf:
            for readline in vplf.readlines():
                line = readline.strip().split()
                if len(line) > 1:
                    if line[0] == "saBodyFiles":
                        for ll in line:
                            if ".in" in ll:
                                self.infile_list.append(ll)
        
        # Parse template unit settings from vpl.in
        self._template_units = self._parse_template_units()
        self._conversion_factors = self._compute_conversion_factors()

        # Fixed parameter substitutions - to be implemented!
        if fixsub is not None:
            self.fixparam = list(fixsub.keys())
            self.fixvalue = list(fixsub.values())

        # Set output timesteps (if specified, otherwise will default to same as dStopTime)
        if timesteps is not None:
            try:
                self.timesteps = timesteps.si.value
            except:
                raise ValueError("Units for timestep not valid.")
        else:
            self.timesteps = None

        # Run model foward (true) or backwards (false)?
        self.forward = forward

        # Set initial simulation time (dAge) in SI seconds
        try:
            self.time_init = time_init.si.value
        except:
            raise ValueError("Units for time_init not valid.")


    def _parse_template_units(self):
        """Read vpl.in and extract the original unit settings."""
        template_units = {}
        vpl_path = os.path.join(self.inpath, self.vplfile)
        with open(vpl_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                key = parts[0]
                if key not in _VPLANET_UNITS:
                    continue
                unit_str = parts[1].lower()
                unit_map = _VPLANET_UNITS[key]
                if unit_str in unit_map:
                    template_units[key] = unit_map[unit_str]
                else:
                    print(f"  Warning: unrecognized {key} value "
                          f"'{parts[1]}', assuming SI")
        return template_units

    def _compute_conversion_factors(self):
        """Compute multiplicative factors from template units to SI."""
        factors = {}
        for key, tmpl_unit in self._template_units.items():
            si_unit = _SI_UNITS[key]
            try:
                factors[key] = (1.0 * tmpl_unit).to(si_unit).value
            except u.UnitConversionError:
                factors[key] = 1.0
        return factors

    def _convert_file_units(self, file_content, optimized_params):
        """Convert non-optimized parameter values from template to SI."""
        lines = file_content.split('\n')
        converted = []
        for line in lines:
            converted.append(
                self._convert_param_line(line, optimized_params)
            )
        return '\n'.join(converted)

    def _convert_param_line(self, line, optimized_params):
        """Convert a single parameter line from template to SI units."""
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            return line
        parts = stripped.split()
        if len(parts) < 2 or not parts[0].startswith('d'):
            return line
        param_name = parts[0]
        if param_name in optimized_params:
            return line
        try:
            value = float(parts[1])
        except ValueError:
            return line
        if value <= 0:
            return line
        unit_key = _get_unit_key(param_name)
        if unit_key is None:
            return line
        factor = self._conversion_factors.get(unit_key, 1.0)
        if factor == 1.0:
            return line
        new_value = value * factor
        idx = line.find(parts[1])
        return line[:idx] + f"{new_value:.10e}" + line[idx + len(parts[1]):]

    @staticmethod
    def _convert_to_si(converted, in_unit):
        """Convert a Quantity to SI.  Negative units are VPLanet sign
        conventions and are left unconverted."""
        if isinstance(in_unit, u.Quantity) and in_unit.value < 0:
            return converted
        try:
            return converted.si
        except (u.UnitConversionError, AttributeError):
            return converted


    def initialize_model(self, theta, outpath=None):
        """
        theta   : (float, list) parameter values, corresponding to self.param

        outpath : (str) path to where model infiles should be written
        """

        # Convert parameter values to SI for the forced-SI template.
        # Dex parameters are un-logged to their physical values.
        # All other units are converted to SI (kg, m, s, rad, K).
        theta_conv = np.zeros(self.ninparam)
        theta_new_unit = []

        for ii in range(self.ninparam):
            if self.in_units[ii] is None:
                theta_conv[ii] = theta[ii]
                theta_new_unit.append(None)
            elif isinstance(self.in_units[ii], u.function.logarithmic.DexUnit):
                new_theta = (theta[ii] * self.in_units[ii]).physical
                theta_conv[ii] = new_theta.value
                theta_new_unit.append(new_theta.unit)
            else:
                converted = theta[ii] * self.in_units[ii]
                converted = self._convert_to_si(
                    converted, self.in_units[ii]
                )
                theta_conv[ii] = converted.value
                theta_new_unit.append(converted.unit)

        if self.verbose:
            print("\nInput:")
            print("-----------------")
            for ii in range(self.ninparam):
                print("%s : %s [%s] (user)   --->   %s [%s] (vpl file)"%(self.inparams[ii],
                    theta[ii], self.in_units[ii], theta_conv[ii], theta_new_unit[ii]))
            print("")

        if not os.path.exists(outpath):
            os.makedirs(outpath)
        
        # format list of input parameters
        param_file_all = np.array([x.split('.')[0] for x in self.inparams])  # e.g. ['vpl', 'primary', 'primary', 'secondary', 'secondary']
        param_name_all = np.array([x.split('.')[1] for x in self.inparams])  # e.g. ['dStopTime', 'dRotPeriod', 'dMass', 'dRotPeriod', 'dMass']

        # format list of output parameters
        key_split  = []
        unit_split = []
        for ii, key in enumerate(self.outparams):
            if "final" in key.split("."):
                key_split.append(key.split(".")[1:])
                unit_split.append([key.split(".")[1], self.out_units[ii]])
        out_name_split = np.array(key_split).T  # e.g. [['primary', 'secondary', 'secondary'], ['RotPer', 'RotPer', 'OrbPeriod']]
        out_unit_split = np.array(unit_split).T # e.g. [['primary', 'secondary', 'secondary'], [Unit("d"), Unit("d"), Unit("d")]]

        # save dictionary for retrieving output arrays
        out_body_name_dict = {key: [] for key in set(out_name_split[0])}
        out_body_unit_dict = {key: [] for key in set(out_unit_split[0])}
        for ii in range(out_name_split.shape[1]):
            out_body_name_dict[out_name_split[0][ii]].append(out_name_split[1][ii])
            out_body_unit_dict[out_unit_split[0][ii]].append(out_unit_split[1][ii])
        self.out_body_name_dict = out_body_name_dict  # e.g. {'secondary': ['RotPer', 'OrbPeriod'], 'primary': ['RotPer']}
        self.out_body_unit_dict = out_body_unit_dict  # e.g. {'secondary': [Unit("d"), Unit("d")], 'primary': [Unit("d")]}

        optimized_param_names = set(param_name_all)

        for file in self.infile_list: # vpl.in, primary.in, secondary.in
            with open(os.path.join(self.inpath, file), 'r') as f:
                file_in = f.read()

            # Convert non-optimized parameters from template units to SI
            file_in = self._convert_file_units(file_in, optimized_param_names)

            ind = np.where(param_file_all == file.strip('.in'))[0]
            theta_file = theta_conv[ind]
            param_name_file = param_name_all[ind]

            # get saOutputOrder for evolution
            if file.strip('.in') in out_name_split[0]:
                output_order_vars = out_name_split[1][np.where(out_name_split[0] == file.strip('.in'))[0]]
                output_order_str = "Time " + " ".join(
                    ["-" + v for v in output_order_vars]
                )

                # Set variables for tracking evolution
                file_in = re.sub("%s(.*?)#" % "saOutputOrder", "%s %s #" % ("saOutputOrder", output_order_str), file_in)
                
            # iterate over all input parameters, and substitute parameters in appropriate files
            for i in range(len(theta_file)):
                file_in = re.sub("%s(.*?)#" % param_name_file[i], "%s %.10e #" % (param_name_file[i], theta_file[i]), file_in)

                # Set output timesteps to simulation stop time
                if param_name_file[i] == 'dStopTime':
                    file_in = re.sub("%s(.*?)#" % "dOutputTime", "%s %.10e #" % ("dOutputTime", theta_file[i]), file_in)

            # if VPL file
            if file == 'vpl.in':
                file_in = re.sub("%s(.*?)#" % "sSystemName", "%s %s #" % ("sSystemName", self.sys_name), file_in)

                # Set units to SI
                file_in = re.sub("%s(.*?)#" % "sUnitMass", "%s %s #" % ("sUnitMass", "kg"), file_in)
                file_in = re.sub("%s(.*?)#" % "sUnitLength", "%s %s #" % ("sUnitLength", "m"), file_in)
                file_in = re.sub("%s(.*?)#" % "sUnitTime", "%s %s #" % ("sUnitTime", "sec"), file_in)
                file_in = re.sub("%s(.*?)#" % "sUnitAngle", "%s %s #" % ("sUnitAngle", "rad"), file_in)
                file_in = re.sub("%s(.*?)#" % "sUnitTemp", "%s %s #" % ("sUnitTemp", "K"), file_in)

                # Set output timesteps (if specified, otherwise will default to same as dStopTime)
                if self.timesteps is not None:
                    file_in = re.sub("%s(.*?)#" % "dOutputTime", "%s %.10e #" % ("dOutputTime", self.timesteps), file_in)

                # Run evolution forward or backward
                if self.forward == True:
                    file_in = re.sub("%s(.*?)#" % "bDoForward", "%s %s #" % ("bDoForward", "1"), file_in)
                    file_in = re.sub("%s(.*?)#" % "bDoBackward", "%s %s #" % ("bDoBackward", "0"), file_in)
                else:
                    file_in = re.sub("%s(.*?)#" % "bDoForward", "%s %s #" % ("bDoForward", "0"), file_in)
                    file_in = re.sub("%s(.*?)#" % "bDoBackward", "%s %s #" % ("bDoBackward", "1"), file_in)

            else: # (not VPL file)
                # Set output timesteps (if specified, otherwise will default to same as dStopTime)
                if self.time_init is not None:
                    file_in = re.sub("%s(.*?)#" % "dAge", "%s %.10e #" % ("dAge", self.time_init), file_in)

            write_file = os.path.join(outpath, file)
            with open(write_file, 'w') as f:
                print(file_in, file = f)

            if self.verbose:
                print(f"Created file {write_file}")


    def get_outparam(self, output, **kwargs):
        """
        output    : (vplot object) results form a model run obtained using vplot.GetOutput()

        outparams : (str, list) return specified list of parameters from log file
                    ['initial.primary.Luminosity', 'final.primary.Radius', 'initial.secondary.Luminosity', 'final.secondary.Radius']
        """
        outvalues = np.zeros(self.noutparam)

        for i in range(self.noutparam):
            base = kwargs.get('base', output.log)
            for attr in self.outparams[i].split('.'):
                base = getattr(base, attr)

            # Apply unit conversions to SI
            if self.out_units[i] is None:
                outvalues[i] = base
            else:
                try:
                    outvalues[i] = base.to(self.out_units[i]).value
                except:
                    outvalues[i] = np.nan

        return outvalues

    
    def get_evol(self, output):

        """
        warning: this is going to break for outparams that aren't formatted final.body.param
        """

        evol_out = []

        for bf in sorted(self.out_body_name_dict.keys()):

            body_outputs = getattr(output, bf)[:]

            # arr[0] is time, create separate array
            time_out = body_outputs[0]
            body_out_array = body_outputs[1:]

            body_out_units = self.out_body_unit_dict[bf]
            body_out_names = self.out_body_name_dict[bf]
            body_nparam = len(self.out_body_unit_dict[bf])

            for ii in range(body_nparam):
                if body_out_units[ii] is None:
                        evol_out.append(body_out_array[ii].value)
                else:
                    try:
                        evol_out.append(body_out_array[ii].to(body_out_units[ii]))
                    except:
                        print(f"Failed to convert parameter {bf} {body_out_names[ii]} to unit {body_out_units[ii]}")
                        print(f"{bf} {body_out_names[ii]} array: {body_out_array[ii]}")

        return time_out, evol_out


    def run_model(self, theta, remove=True, outsubpath=None, return_output=False):
        """
        theta     : (float, list) parameter values, corresponding to self.inparams

        remove    : (bool) True will erase input/output files after model is run
        """

        # randomize output directory
        if outsubpath is None:
            # create a unique randomized path to execute the model files
            outsubpath = random.randrange(16**12)
        outpath = f"{self.outpath_base}/{outsubpath}"

        self.initialize_model(theta, outpath=outpath)
        
        t0 = time.time()

        # Execute the model!
        subprocess.call([f"{self.executable} {self.vplfile}"], cwd=outpath, shell=True)

        try:
            output = vplanet.get_output(outpath)
        except Exception:
            if remove and os.path.isdir(outpath):
                shutil.rmtree(outpath)
            return np.full(len(self.outparams), np.nan)

        if return_output == True:
            return output

        if self.verbose == True:
            print('Executed model %s/vpl.in %.3f s'%(outpath, time.time() - t0))

        # return final values
        outvalues = self.get_outparam(output)
        model_final = outvalues

        # if timesteps are specified, return evolution
        if self.timesteps is not None:
            model_time, evol = self.get_evol(output)
            model_evol = dict(zip(self.outparams, evol))
            model_evol['Time'] = model_time.to(u.yr)

        if self.verbose:
            print("\nOutput:")
            print("-----------------")
            for ii in range(self.noutparam):
                print("%s : %s [%s]"%(self.outparams[ii], model_final[ii], self.out_units[ii]))
            print("")

        if remove == True:
            shutil.rmtree(outpath)

        if self.timesteps is None:
            return model_final
        else:
            return model_evol


    def quickplot_evol(self, time, evol, ind=None):

        import matplotlib.pyplot as plt
        from matplotlib import rc
        try:
            rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
            rc('text', usetex=True)
            rc('xtick', labelsize=20)
            rc('ytick', labelsize=20)
        except:
            rc('text', usetex=False)

        time = np.array(time)
        evol = np.array(evol)

        if ind is None:
            ind = np.arange(len(evol))
        evol = evol[ind]
        nplots = len(evol)

        fig, axs = plt.subplots(nplots, 1, figsize=[4*nplots, 8], sharex=True)
        for ii in range(nplots):
            axs[ii].plot(time, evol[ii])
        plt.close()

        return fig
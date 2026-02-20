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

        # Set initial simulation time (dAge)
        try:
            self.time_init = time_init.si.value
        except:
            raise ValueError("Units for time_init not valid.")


    def initialize_model(self, theta, outpath=None):
        """
        theta   : (float, list) parameter values, corresponding to self.param

        outpath : (str) path to where model infiles should be written
        """
        
        # Convert parameter values to VPLanet-compatible units.
        # Dex parameters are un-logged to their physical values.
        # All other parameters are passed through unchanged, so the
        # config units must match VPLanet's expected input units.
        theta_conv = np.zeros(self.ninparam)
        theta_new_unit = []

        for ii in range(self.ninparam):
            if self.in_units[ii] is None:
                theta_conv[ii] = theta[ii]
                theta_new_unit.append(None)
            elif isinstance(self.in_units[ii], u.function.logarithmic.DexUnit):
                # un-log dex parameters to physical values
                new_theta = (theta[ii] * self.in_units[ii]).physical
                theta_conv[ii] = new_theta.value
                theta_new_unit.append(new_theta.unit)
            else:
                theta_conv[ii] = theta[ii]
                theta_new_unit.append(self.in_units[ii])

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

        for file in self.infile_list: # vpl.in, primary.in, secondary.in
            with open(os.path.join(self.inpath, file), 'r') as f:
                file_in = f.read()
                
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
                file_in = re.sub("%s*" % "saOutputOrder", "%s %s #" % ("saOutputOrder", output_order_str), file_in)
                
            # iterate over all input parameters, and substitute parameters in appropriate files
            for i in range(len(theta_file)):
                file_in = re.sub("%s(.*?)#" % param_name_file[i], "%s %.10e #" % (param_name_file[i], theta_file[i]), file_in)

                # Set output timesteps to simulation stop time
                if param_name_file[i] == 'dStopTime':
                    file_in = re.sub("%s(.*?)#" % "dOutputTime", "%s %.10e #" % ("dOutputTime", theta_file[i]), file_in)

            # if VPL file
            if file == 'vpl.in':
                file_in = re.sub("%s(.*?)#" % "sSystemName", "%s %s #" % ("sSystemName", self.sys_name), file_in)

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
        except:
            print("If no logfile is found, it's probably because there was something wrong with the infile formatting.")
            print(f"Try executing 'vplanet vpl.in' in directory {outpath} to diagnose the error in vplanet.")

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
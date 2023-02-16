#!/usr/bin/env python
# coding: utf-8

# In[1]:




import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from itertools import repeat


from snewpy import snowglobes
from astropy import units as u
from snewpy.neutrino import Flavor, MassHierarchy, MixingParameters
from snewpy.models.ccsn import *
from snewpy.flavor_transformation import AdiabaticMSW


SNOwGLoBES_path = "/home/aklima/Desktop/snowglobes"
SNEWPY_models_base = "/home/aklima/Desktop/SNEWPY_models"  # directory containing SNEWPY models
# set parameters that don't vary over all models
# set distance in kpc
distance = 1.0

# set SNOwGLoBES detector to use
detector = "ar40kt"

# set desired flavor transformation
transformation = 'NoTransformation'

# set SAME times as the ones you ran in the simulation - these are my higher resolution ones but I overwrite later
window_tstart = 0.51 # Fornax adjusted time bins
window_tend = 1.9
window_bins = 180 # .00772s time bin

window_tstart_burst = 0.01 # Fornax adjusted time bins
window_tend_burst = 0.50
window_bins_burst = 200 # .00245 s time bin

tstart = np.linspace(window_tstart, window_tend, window_bins, endpoint=False) 
tend = tstart + (window_tend - window_tstart) / window_bins 
tmid = (tstart + tend) * 0.5 

tstart_burst = np.linspace(window_tstart_burst, window_tend_burst, window_bins_burst, endpoint=False) 
tend_burst = tstart_burst + (window_tend_burst - window_tstart_burst) / window_bins_burst 
tmid_burst = (tstart_burst + tend_burst) * 0.5 

tstart = np.append(tstart_burst, tstart)
tend = np.append(tend_burst, tend)

# rename factor
change_to_s = window_bins / (window_tend - window_tstart) / 100
change_to_s_burst = window_bins_burst / (window_tend_burst - window_tstart_burst) / 100

# number of lines in dat file
num_lines = 500

plt.rcParams.update({'font.size': 18})

# This block is not necessary. Later, I use the `pcolormesh` function - just set `cmap = "viridis"` or something else
# make new colorbar

# opening the file in read mode
new_colorbar = open("/home/aklima/Desktop/Lima/colormap_updated.txt", "r")
  
# reading the file
data = new_colorbar.read()
  
# replacing end splitting the text 
# when newline ('\n') is seen.
c = data.split('\n')

new_colorbar.close()

cm_u = mpl.colors.ListedColormap(c)

# big function just to read data from SNEWPY files
"""Args:
energy_inst: long 1-by-x array of energy at each line in each file - I provide blank array
fluence_inst: long 1-by-x array of specified fluence flavor data at each line in each file - I provide blank array
nu_x_fluence: 1-by-x array of specified fluence flavor data at each time-step (each file) - I provide blank array
flavor: You specify "nux", "nue", "anue", or "all"
file_directory: path to where you put your files. Can only contain files from SNEWPY

"""

def get_fluence_and_energy_arrays(energy_inst, fluence_inst, nu_x_fluence, flavor, file_directory):
    # reads location you provide
    sorted_dir = os.listdir(file_directory)
    # sorts these files in numerical order
    sorted_dir = sorted(sorted_dir, key=lambda x:float(re.findall("(\d+)", x)[-1]))
    # iterates through files, opening them to read
    for filename in sorted_dir:
        with open(os.path.join(file_directory, filename), 'r') as f:
            nu_x_sum = 0
            next(f)
            next(f) # skip string headers
            next(f) # skip 0 energy bin
            if flavor == "nux":
                for line in f:
                    nu_x_sum += (float(line.split()[2]) + float(line.split()[3]) + float(line.split()[5]) + float(line.split()[6]))
                    energy_inst.append(float(line.split()[0]))
                    fluence_inst.append(float(line.split()[2]) + float(line.split()[3]) + float(line.split()[5]) + float(line.split()[6]))
                nu_x_fluence.append(nu_x_sum)
            if flavor == "nue":
                for line in f:
                    nu_x_sum += float(line.split()[1])
                    energy_inst.append(float(line.split()[0]))
                    fluence_inst.append(float(line.split()[1]))
                nu_x_fluence.append(nu_x_sum)
            if flavor == "anue":
                for line in f:
                    nu_x_sum += float(line.split()[4])
                    energy_inst.append(float(line.split()[0]))
                    fluence_inst.append(float(line.split()[4]))
                nu_x_fluence.append(nu_x_sum)
            if flavor == "all":
                for line in f:
                    nu_x_sum += (float(line.split()[1]) + float(line.split()[2]) + float(line.split()[3]) + float(line.split()[4]) + float(line.split()[5]) + float(line.split()[6]))
                    energy_inst.append(float(line.split()[0]))
                    fluence_inst.append(float(line.split()[1]) + float(line.split()[2]) + float(line.split()[3]) + float(line.split()[4]) + float(line.split()[5]) + float(line.split()[6]))
                nu_x_fluence.append(nu_x_sum)

# sorts time steps into different phases, for graphing and resolution purposes
def sort_by_phase(tstart, window_tend_burst):
    high_res = []
    for i in tstart:
        if i <= window_tend_burst:
            high_res.append(i)
    
    burst, accretion, cooling = [], [], []
    for i in (tstart):
        if i <= 0.06:   #0.1
            burst.append(i)
        if i > 0.06 and i <= 0.5:
            accretion.append(i)
        if i > .5: #i > .5:
            cooling.append(i)
    return burst, accretion, cooling, high_res

# for the two-dimension plots, this function extends the 1-by-x arrays into x-by-num-lines array to match 
# your input data
def extend_phase(burst, accretion, cooling):
    ext_burst = [x for item in burst for x in repeat(item, num_lines)]
    ext_burst = np.array_split(ext_burst,len(burst))

    ext_accretion = [x for item in accretion for x in repeat(item, num_lines)]
    ext_accretion = np.array_split(ext_accretion,len(accretion))

    ext_cooling = [x for item in cooling for x in repeat(item, num_lines)]
    ext_cooling = np.array_split(ext_cooling,len(cooling))
    
    return ext_burst, ext_accretion, ext_cooling

# this function just splits the long fluence and energy arrays into the right shape - times on x and fluence or energy on y
def shape_and_scale_fluence_energy(fluence_test, energy_test, high_res):
    l = np.array_split(fluence_test,window_bins + window_bins_burst)
    e = np.array_split(energy_test,window_bins + window_bins_burst)
    # changes to MeV
    e = [value * 1000 for value in e]
    
    return l, e

# this function takes the higher resolution part and changes it to seconds or bins 
def scale_fluence(l, high_res, change_to_s_burst, change_to_s):
    l = [value * change_to_s_burst for value in l[:len(high_res)-1]] + [value * change_to_s for value in l[len(high_res)-1:]]
    return l
    
# makes flavors look pretty    
def formatted_flavor(flavor):
    if flavor == "nux":
        overleaf_flavor = "$\\nu_x$"
    if flavor == "anue":
        overleaf_flavor = "$\\overline{\\nu_e}$"
    if flavor == "nue":
        overleaf_flavor = "$\\nu_e$"
    if flavor == "all":
        overleaf_flavor = "$\\nu_e$, $\\overline{\\nu_e}$, and $\\nu_x$"
    return overleaf_flavor

# plots integrated version with matplotlib
def plot_projected(tstart, nue_test, ax1, ax2, ax3, local_max_location, overleaf_flavor, burst, accretion, cooling):
    total = len(tstart)
    ax1.plot(tstart[:len(burst)] * u.s, nue_test[local_max_location:len(burst) + local_max_location], label = overleaf_flavor, drawstyle='steps-mid')
    ax2.plot(tstart[len(burst):len(burst)+len(accretion)] * u.s, nue_test[len(burst) + local_max_location:len(burst)+len(accretion) + local_max_location], label = overleaf_flavor, drawstyle='steps-mid')
    ax3.plot(tstart[len(burst) + len(accretion):total -local_max_location] * u.s , nue_test[len(burst) + len(accretion) + local_max_location:total -local_max_location], label = overleaf_flavor, drawstyle='steps-mid')

# normalizes by max
def normalize_by_peak(test13, burst, relative_to):
    peak_first = np.max(relative_to[:len(burst)])
    peak_second = np.max(relative_to[len(burst):])
    list1 = [value/peak_first for value in test13[:len(burst)]] 
    list2 = [value/(peak_first * 10) for value in test13[len(burst):]]
    list1.extend(list2)
    return list1

# special function to change resolution to ms 
def preserve_event_rates(test13, burst, relative_to):
    list1 = [value * change_to_s_burst for value in test13[:len(burst)-1]] 
    list2 = [value * change_to_s for value in test13[len(burst)-1:]]
    list1.extend(list2)
    return list1

# finds location of local max in array to set peak at 0
def find_local_max(all_test, high_res, adjust):
    local_max_location = np.argmax(all_test[:len(high_res)]) - adjust
    return local_max_location

def get_data_and_plot_fluence(flavor, files, mass, model, msw, projection, tstart, window_tend_burst):
    
    all_for_max, test, test1 = [], [], []
    energy_test, fluence_test, nu_x_test = [], [], []
    
    
    get_fluence_and_energy_arrays(test, test1, all_for_max, "all", files)  
    get_fluence_and_energy_arrays(energy_test, fluence_test, nu_x_test, flavor, files)
    
    
    burst, accretion, cooling, high_res = sort_by_phase(tstart, window_tend_burst)
    ext_burst, ext_accretion, ext_cooling = extend_phase(burst, accretion, cooling)
    
    
    # shape fluence
    l, e = shape_and_scale_fluence_energy(fluence_test, energy_test, high_res)
    overleaf_flavor = formatted_flavor(flavor)

    
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 5), tight_layout=True)
    if projection == "integrated":
        nu_x_test = scale_fluence(nu_x_test, high_res, change_to_s_burst, change_to_s)
        local_max = np.max(nu_x_test[:len(high_res) - 1])
        local_max_location = find_local_max(nu_x_test, high_res)

        
        plot_projected(nu_x_test, ax1, ax2, ax3, local_max_location, overleaf_flavor, burst, accretion, cooling)
        ax1.set_ylim(0,local_max * 1.1)
        ax1.margins(0)
        ax2.set_ylim(0,local_max * 1.1)
        ax2.margins(0)
        ax3.set_ylim(0,local_max * 1.1)
        ax3.margins(0)
        ax1.set_ylabel(f"{overleaf_flavor} events per ms")
        
        
    if projection == None:
        # scale the colorbars to be the same for each subplot
        l = scale_fluence(l, high_res, change_to_s_burst, change_to_s)
        local_max = np.max(l[:len(burst) - 1])
        local_min = np.min(l[:len(burst) - 1])
        local_max_location = 4#np.argmax(all_for_max[:len(high_res) - 1]) - 25
        
        pcm1 = ax1.pcolormesh(ext_burst, e[local_max_location:len(burst) + local_max_location], l[local_max_location:len(burst) + local_max_location])#, vmin = local_min, vmax = local_max) ## these change limits of colorbars - set all to be the same
        pcm2 = ax2.pcolormesh(ext_accretion, e[local_max_location + len(burst):len(accretion) + len(burst) + local_max_location], l[len(burst) + local_max_location:len(accretion) + len(burst) + local_max_location])#, vmin = local_min, vmax = local_max)
        pcm3 = ax3.pcolormesh(ext_cooling[:-local_max_location], e[len(accretion) + len(burst) + local_max_location:], l[len(accretion) + len(burst) + local_max_location:])#, cmap = cm_u, vmin = local_min, vmax = local_max)

        ax1.set_ylim(0,50)
        ax2.set_ylim(0,50)        
        ax3.set_ylim(0,50)
        ax1.set_ylabel("Energy (MeV)")
        projection = ""
    
    ax1.set_xlabel("Burst phase (t[s])")
    ax2.set_xlabel("Accretion phase (t[s])")
    ax2.set_yticks([])
    ax3.set_xlabel("Cooling phase (t[s])")
    ax3.set_yticks([]) 
    
    plt.suptitle(f'{projection}{overleaf_flavor} fluence from {model} from a {mass} M☉ progenitor, 1kpc away - {msw} effects')
    plt.savefig(f'{projection}_{flavor}_{mass}_{model}_{msw}.png')
    plt.show()
    
 
    if projection == None:
    # draw a new figure and replot the colorbar there
        fig,ax = plt.subplots(figsize=(22,3))
        cbar_cooling = plt.colorbar(pcm1,ax=ax)
        ax.remove()
        cbar_cooling.set_label(f'Fluence from {overleaf_flavor} per ms')
        plt.savefig(f'{flavor}_{mass}_{model}_{msw}_cbar.png',bbox_inches='tight')
    

window_tstart = 0.51 # Fornax adjusted time bins
window_tend = 1.9
window_bins = 180 # .1 s time bin

window_tstart_burst = 0.01 # Fornax adjusted time bins
window_tend_burst = 0.50
window_bins_burst = 200 # .01 s time bin

tstart = np.linspace(window_tstart, window_tend, window_bins, endpoint=False) 
tend = tstart + (window_tend - window_tstart) / window_bins 
tmid = (tstart + tend) * 0.5 

tstart_burst = np.linspace(window_tstart_burst, window_tend_burst, window_bins_burst, endpoint=False) 
tend_burst = tstart_burst + (window_tend_burst - window_tstart_burst) / window_bins_burst 
tmid_burst = (tstart_burst + tend_burst) * 0.5 

tstart = np.append(tstart_burst, tstart)
tend = np.append(tend_burst, tend)



my_files = "/home/aklima/Desktop/Lima/fluence/Fornax_2021_lum_spec_12M_r10000_dat.h5_AdiabaticMSW_NMO"
get_data_and_plot_fluence("nue", my_files, "12", "Fornax", "NMO MSW", None, tstart, window_tend_burst)

def get_data_and_plot_all_int_fluence(files, mass, model, msw, tstart):
    nu_x_test, anue_test, nue_test, energy_test, fluence_test, all_test = [], [], [], [], [], []
    
    get_fluence_and_energy_arrays(energy_test, fluence_test, anue_test, "anue", files)
    get_fluence_and_energy_arrays(energy_test, fluence_test, nue_test, "nue", files)
    get_fluence_and_energy_arrays(energy_test, fluence_test, nu_x_test, "nux", files)
    get_fluence_and_energy_arrays(energy_test, fluence_test, all_test, "all", files)
    

    
    burst, accretion, cooling, high_res = sort_by_phase(tstart, window_tend_burst)
    ext_burst, ext_accretion, ext_cooling = extend_phase(burst, accretion, cooling)
    
    anue_test = preserve_event_rates(anue_test, high_res, all_test)
    nue_test = preserve_event_rates(nue_test, high_res, all_test)
    nu_x_test = preserve_event_rates(nu_x_test, high_res, all_test)
    all_test = preserve_event_rates(all_test, high_res, all_test)
    
    # shape fluence
    l, e = shape_and_scale_fluence_energy(fluence_test, energy_test, high_res)
    local_max = np.max(all_test[:len(burst) - 1])

    local_max_location = 0#np.argmax(all_test[:len(high_res) - 1]) - 5

   # tstart = [x-.03 for x in tstart]
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 5), tight_layout=True)
    plt.suptitle(f'Fluence by flavor from {model} model from a {mass} M☉ progenitor, 1kpc away - {msw} effects')
    
    plot_projected(tstart, nue_test, ax1, ax2, ax3, local_max_location, formatted_flavor("nue"), burst, accretion, cooling)
    plot_projected(tstart, anue_test, ax1, ax2, ax3, local_max_location, formatted_flavor("anue"), burst, accretion, cooling)
    plot_projected(tstart, nu_x_test, ax1, ax2, ax3, local_max_location, formatted_flavor("nux"), burst, accretion, cooling)
    plot_projected(tstart, all_test, ax1, ax2, ax3, local_max_location, formatted_flavor("all"), burst, accretion, cooling)
    

    ax1.set_xlabel("Burst phase (t[s])")
    ax1.set_ylabel(f"Events per 10 ms")
    ax1.set_ylim(0,local_max * 1.1)
    ax1.margins(0)
    #ax1.legend(prop={'size': 17})

    ax2.set_xlabel("Accretion phase (t[s])")
    ax2.set_yticks([])
    ax2.set_ylim(0,local_max * 1.1)
    ax2.margins(0)
    ax2.legend(prop={'size': 17})

    ax3.set_xlabel("Cooling phase (t[s])")
    ax3.set_yticks([])
    #ax3.set_ylim(0,local_max * 1.1)
    ax3.set_ylim(0,local_max * 1.1)
    ax3.margins(0)
    #ax3.legend(prop={'size': 17})
    plt.savefig(f'int_all_{mass}_{model}_{msw}.png')
    plt.show()
    

# set SAME times as the ones you ran in the simulation - these are my hig2her resolution ones but I overwrite later
window_tstart = 0.51 # Fornax adjusted time bins
window_tend = 1.9
window_bins = 180 # .005 s time bin

window_tstart_burst = 0.01 # Fornax adjusted time bins
window_tend_burst = 0.50
window_bins_burst = 200 # .002 s time bin

tstart = np.linspace(window_tstart, window_tend, window_bins, endpoint=False) 
tend = tstart + (window_tend - window_tstart) / window_bins 
tmid = (tstart + tend) * 0.5 

tstart_burst = np.linspace(window_tstart_burst, window_tend_burst, window_bins_burst, endpoint=False) 
tend_burst = tstart_burst + (window_tend_burst - window_tstart_burst) / window_bins_burst 
tmid_burst = (tstart_burst + tend_burst) * 0.5 

tstart = np.append(tstart_burst, tstart)
tend = np.append(tend_burst, tend)

my_files = "/home/aklima/Desktop/Lima/fluence/Fornax_2021_lum_spec_12M_r10000_dat.h5_AdiabaticMSW_NMO"
get_data_and_plot_all_int_fluence(my_files, "12", "Fornax", "NMO MSW", tstart)

 

 


# In[ ]:





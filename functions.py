import os, sys
import numpy as np
import scipy.constants
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
from subprocess import Popen, PIPE
import subprocess, shlex
from time import gmtime, strftime
import math
from matplotlib import path
import scipy.integrate as integrate
import multiprocessing as mp
import time
# import argparse
import re
from scipy import stats
from numba import jit
from scipy.spatial import cKDTree


# Fonts

alignment = {'horizontalalignment': 'center', 'verticalalignment': 'baseline'}
families = ['serif', 'sans-serif', 'cursive', 'fantasy', 'monospace']
styles = ['normal', 'italic', 'oblique']
font = FontProperties()

#  tick fonts
font1 = font.copy()
font1.set_family('sans-serif')
font1.set_style('normal')

# label fonts
font2 = font.copy()
font2.set_family('sans-serif')
font2.set_style('normal')
font2.set_weight('normal')
plt.rcParams['mathtext.default'] = 'regular'
plt.rcParams['axes.linewidth'] = 6
bbox = dict(boxstyle="round", fc="0.8")
cmaps = [('Perceptually Uniform Sequential', [
            'viridis', 'plasma', 'inferno', 'magma']),
         ('Sequential', [
            'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']),
         ('Sequential (2)', [
            'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
            'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
            'hot', 'afmhot', 'gist_heat', 'copper']),
         ('Diverging', [
            'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
            'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']),
         ('Qualitative', [
            'Pastel1', 'Pastel2', 'Paired', 'Accent',
            'Dark2', 'Set1', 'Set2', 'Set3',
            'tab10', 'tab20', 'tab20b', 'tab20c']),
         ('Miscellaneous', [
            'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
            'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'hsv',
            'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar'])]


def parameters(input_file, setting, param):
    if os.path.exists(input_file):
        with open(input_file, 'r') as param_input:
            for line_nr, line in enumerate(param_input.readlines()):
                if not line.startswith('#'):
                    line_sep = line.split('#')[0].split('=') 
                    variable = line_sep[0].strip()
                    if variable in setting:
                        param[variable] = split_setting(variable, line_sep[-1])
    else:
        sys.exit('cannot find parameter file')
    return param

def check_variable(setting, param):
    for var in setting:
        if var not in param:
            sys.exit('The variable '+var+' is missing')


def split_setting(variable, setting):
    string_variables = ['pdb_output', 'fes', 'picture_file', 'bulk_values', 'CV1', 'CV2', 'prefix', '1d_location', 'HILLS_skip', 'HILLS_sort', 'HILLS']
    int_variables = ['start', 'end']
    float_variables = ['plot_step','plot_labels', 'plot_interval_x', 'plot_interval_y','plot_energy_min', 'plot_energy_max', 
                        'plot_colour_bar_tick', 'cv_x_interval', 'cv_y_interval', 'cv_labels_size', 
                        'cv_title_height', 'cv_tick_size', 'boot_label_y','boot_labels', 'boot_title_height',
                        'boot_labels_size', 'boot_energy_max', 'boot_step_yaxis',   'converge_labels', 
                        'converge_title_height', 'converge_x_interval', 'equilibration', 'stride','step_xaxis',
                        'step_yaxis', 'bulk_outline_shrink', 'bulk_area']
    complex_variables = ['start_points', 'end_points', 'circle_centers','ellipse_centers']
    list_variables = ['picture_loc', 'cv_min', 'cv_max', 'circle_area', 'ellipse_height', 'ellipse_width', 'ellipse_angle','ref_min', 
                      'ref_max', 'walker_range', 'search_width', 'trim', 'plot_trim', 'pdb_offset']
    T_F_variables = ['circle_plot','ellipse_plot', 'picture', '1d', 'bulk_outline', 'hills_trim']
    if variable in string_variables:
        return str(setting.strip())
    elif variable in int_variables:
        return int(setting.strip()) 
    elif variable in float_variables:
        return float(setting.strip())    
    elif variable in T_F_variables:
        if setting.strip().lower() in ['true','1', 't', 'y', 'yes']:
            return True
        else:
            return False
    elif variable in list_variables:
        return np.array([float(x) for x in setting.strip().split(',')])
    elif variable in complex_variables:
        l = np.array([float(i) for x in setting.strip().split() for i in x.split(',')])
        return l.reshape(int(len(l)/2), 2)
    else:
        sys.exit('The variable '+variable+' is not correct')


def gromacs(cmd):
    print('\nrunning gromacs: \n '+cmd)
    output = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    err, out = output.communicate()
    exitcode = output.returncode
    out=out.decode("utf-8")
    checks = open('gromacs_outputs_out', 'a')
    checks.write(out)
    return out

def ave(a, n) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

def read_bulk(files,time, dx, dy,gau):
    sorted_X_s,sorted_Y_s,sorted_X_l, sorted_Y_l=np.array([]),np.array([]),np.array([]),np.array([])
    for line in open(files, 'r').readlines():
        sorted_X_s= np.append(sorted_X_s, float(line.split()[0]))
        sorted_Y_s= np.append(sorted_Y_s, float(line.split()[1]))
        sorted_X_l= np.append(sorted_X_l, float(line.split()[2]))
        sorted_Y_l= np.append(sorted_Y_l, float(line.split()[3]))
    shuffledxy = np.stack((dx,dy), axis=-1)
    bulk, time_bulk= np.array([]),np.array([])
    for bulk_coord in range(len(sorted_X_l)-1):
        p = path.Path([(sorted_X_l[bulk_coord], sorted_Y_l[bulk_coord]), (sorted_X_l[bulk_coord+1], sorted_Y_l[bulk_coord+1]), (sorted_X_s[bulk_coord+1],sorted_Y_s[bulk_coord+1]), (sorted_X_s[bulk_coord],sorted_Y_s[bulk_coord])])
        bulk_values =  p.contains_points(shuffledxy)
        time_bulk = np.append(time_bulk, time[bulk_values])
        bulk = np.append(bulk, gau[bulk_values])
    time_bulk_index=np.argsort(time_bulk)
    time_bulk= time_bulk[time_bulk_index]
    bulk=bulk[time_bulk_index]      
    return time_bulk, bulk

def readhills(files):
    time, dx, dy, gau=[],[],[],[]
    wtime, wdx, wdy, wgau, tstamp= [],[],[],[],[]
    count=0
    check=False
    with open(files, 'r') as hills_file_in:
        for line in hills_file_in:
            if not line[0] in ['#', '@']:
                    count+=1
                    time.append(float(count)/1000)
                    dx.append(float(line.split()[1]))
                    dy.append(float(line.split()[2]))
                    gau.append(float(line.split()[5]))
    return np.array(time),np.array(dx),np.array(dy),np.array(gau)

def skip_hills(files, skip_out):
    time, dx, dy, gau=[],[],[],[]
    count=0
    with open(skip_out, 'w') as HILLS_skipped:
        with open(files, 'r') as hills_file_in:     
            for line in hills_file_in:
                if not line[0] in ['#', '@']:
                    count+=1
                    if (float(count)/1000).is_integer():        
                        HILLS_skipped.write(line)
                        print('time = ',np.round((float(count)/1000)/1000, 3),' us', end='\r')


def write_file(CV,d1x,d1y,energy,site):
        with open('1D_landscape_site-'+str(site), 'w') as landscape:
            for line in range(len(CV)):
                landscape.write(str(CV[line])+'\t'+str(energy[line])+'\t'+str(d1x[line])+'\t'+str(d1y[line])+'\n')

def readfes(files, param):
    x, y, z, e=[],[],[],[]
    start_fes=time.time()
    cut = []
    for line in open(files, 'r').readlines():
            if not line[0] in ['#', '@'] and len(line)>1:
                try:
                    line_sep = line.split()
                    if param['trim'][0] < float(line_sep[0]) < param['trim'][1] and param['trim'][2] < float(line_sep[1]) < param['trim'][3]:
                        x.append(float(line_sep[0]))
                        y.append(float(line_sep[1]))
                        z.append(float(line_sep[2]))
                        e.append(float(line_sep[3]))
                except:
                    print('error in :', line)
    print('reading in '+files.split('.')[0]+' took:', np.round(time.time() - start_fes, 2))
    return np.array(x),np.array(y),np.array(z),np.array(e)

def clockwiseangle_and_distance(point):
    origin = [0,0]
    refvec = [0, 1]
    vector = [point[0]-origin[0], point[1]-origin[1]]
    lenvector = math.hypot(vector[0], vector[1])
    if lenvector == 0:
        return -math.pi, 0
    normalized = [vector[0]/lenvector, vector[1]/lenvector]
    dotprod  = normalized[0]*refvec[0] + normalized[1]*refvec[1]     
    diffprod = refvec[1]*normalized[0] - refvec[0]*normalized[1]     
    angle = math.atan2(diffprod, dotprod)

    if angle < 0:
        return 2*math.pi+angle, lenvector
    return angle, lenvector

def bulk_val(x,y,z, param): 
        bulk =[]
        shuffledxy=[]
        bulk_x,bulk_y=np.array([]),np.array([])
        bulk_ring = np.where(np.logical_and(-0.002< z, z < 0))
        bulk_x, bulk_y=x[bulk_ring], y[bulk_ring]
        shuffledxy = np.stack((x,y), axis=-1)
        # print('done outline')
        centered=[]
        bulk_x_new,bulk_y_new=bulk_x, bulk_y
        for i in range(len(bulk_x)):
                x_ind, y_ind= bulk_x[i], bulk_y[i]
                center=np.where(np.sqrt(((x_ind-bulk_x_new)**2)+((y_ind-bulk_y_new)**2)) < 0.1)
                to_del=np.where(np.sqrt(((x_ind-bulk_x_new)**2)+((y_ind-bulk_y_new)**2)) < 0.2)
                if len(center[0])> 0 :
                    centered.append([np.mean(bulk_x_new[center]),np.mean(bulk_y_new[center])])              
                    bulk_x_new = np.delete(bulk_x_new,to_del)
                    bulk_y_new = np.delete(bulk_y_new,to_del)
        sorted_coord=sorted(centered, key=clockwiseangle_and_distance)
        run=False
        for i in range(1, len(sorted_coord)): 
            if np.sqrt(((sorted_coord[i-1][0]-0)**2)+((sorted_coord[i-1][1]-0)**2)) > 2 and not run:
                run=True
                sorted_x, sorted_y = [sorted_coord[i-1][0]], [sorted_coord[i-1][1]]
            if run:
                if np.sqrt(((sorted_coord[i][0]-sorted_x[-1])**2)+((sorted_coord[i][1]-sorted_y[-1])**2)) < 0.5 and run:
                    sorted_x.append(sorted_coord[i][0])
                    sorted_y.append(sorted_coord[i][1])
        sorted_x.append(sorted_x[0])
        sorted_y.append(sorted_y[0])


    #### gives line around outside
        xy=np.stack((sorted_x,sorted_y), axis=-1)
        sorted_X_l,sorted_Y_l,sorted_X_s, sorted_Y_s  = np.array([]),np.array([]),np.array([]),np.array([])
        for coord in xy:
            s=1
            l=0
            while l <= param['bulk_outline_shrink']:   ### max distance from outer line 
                s-=0.01
                coords=[np.round(coord[0]*s, 3),np.round(coord[1]*s, 3)]
                l = np.sqrt(((coord[0]-coords[0])**2)+((coord[1]-coords[1])**2))
                if l <= param['bulk_outline_shrink']-param['bulk_area']:   ### min distance form outer line 
                    low_coords=coords
            sorted_X_l = np.append(sorted_X_l, coords[0])
            sorted_Y_l = np.append(sorted_Y_l, coords[1])
            sorted_X_s = np.append(sorted_X_s, low_coords[0])
            sorted_Y_s = np.append(sorted_Y_s, low_coords[1])
        with open('bulk_values', 'w') as bulk_values:
            for i in range(len(sorted_X_l)):
                bulk_values.write(str(sorted_X_s[i])+'\t'+str(sorted_Y_s[i])+'\t'+str(sorted_X_l[i])+'\t'+str(sorted_Y_l[i])+'\n')
        bulk, sorted_X_s, sorted_Y_s, sorted_X_l, sorted_Y_l = find_bulk(x,y,z,shuffledxy,sorted_X_s, sorted_Y_s, sorted_X_l, sorted_Y_l)
        return bulk, sorted_X_s, sorted_Y_s, sorted_X_l, sorted_Y_l


def find_bulk(x,y,z,shuffledxy, sorted_X_s, sorted_Y_s, sorted_X_l, sorted_Y_l):
###find values inbetween points
    bulk= np.array([])
    for bulk_coord in range(len(sorted_X_l)-1):
        x_high= np.max([sorted_X_l[bulk_coord],sorted_X_s[bulk_coord],sorted_X_l[bulk_coord+1],sorted_X_s[bulk_coord+1]])
        x_low = np.min([sorted_X_l[bulk_coord],sorted_X_s[bulk_coord],sorted_X_l[bulk_coord+1],sorted_X_s[bulk_coord+1]])
        y_high= np.max([sorted_Y_l[bulk_coord],sorted_Y_s[bulk_coord],sorted_Y_l[bulk_coord+1],sorted_Y_s[bulk_coord+1]])
        y_low = np.min([sorted_Y_l[bulk_coord],sorted_Y_s[bulk_coord],sorted_Y_l[bulk_coord+1],sorted_Y_s[bulk_coord+1]])
        p = path.Path([(sorted_X_l[bulk_coord], sorted_Y_l[bulk_coord]), (sorted_X_l[bulk_coord+1], sorted_Y_l[bulk_coord+1]), (sorted_X_s[bulk_coord+1],sorted_Y_s[bulk_coord+1]), (sorted_X_s[bulk_coord],sorted_Y_s[bulk_coord])])
        square=np.where( np.logical_and(np.logical_and(x_low < x, x < x_high),np.logical_and(y_low < y, y < y_high)))
        bulk_values =  p.contains_points(shuffledxy[square])
        bulk = np.append(bulk, z[square][bulk_values])
    if len(bulk) == 0:
        bulk=0
    bulk = np.round(float(np.nanmean(bulk)),2)
    return bulk, sorted_X_s, sorted_Y_s, sorted_X_l, sorted_Y_l

def trim_hills(files, time, dx, dy, gau):
    print('trimming HILLS')
    sorted_X_s,sorted_Y_s,sorted_X_l, sorted_Y_l=np.array([]),np.array([]),np.array([]),np.array([])
    for line in open(files, 'r').readlines():
        sorted_X_s= np.append(sorted_X_s, float(line.split()[0])*1.1)
        sorted_Y_s= np.append(sorted_Y_s, float(line.split()[1])*1.1)
    hillxy = np.stack((dx,dy), axis=-1)
    bulkxy = np.stack((sorted_X_s,sorted_Y_s), axis=-1)
    p = path.Path(bulkxy)
    trimmed_values =  p.contains_points(hillxy)
    return time[trimmed_values], dx[trimmed_values], dy[trimmed_values], gau[trimmed_values]

def hills_converge(param, show):
    lab=[]
    time, dx, dy, gau = readhills(param['HILLS_skip'])
    if param['hills_trim']:
        time, dx, dy, gau = trim_hills(param['bulk_values'],time, dx, dy,gau)
    time_bulk ,bulk = read_bulk(param['bulk_values'],time, dx, dy,gau)
    plot_numbers=2
    if not param['circle_plot'] and not param['ellipse_plot']:
        plt.figure(1, figsize=(20,10))
    if param['circle_plot']:
        plot_numbers+=len(param['circle_centers'])
        plt.figure(1, figsize=(20,30))
    if param['ellipse_plot']:
        plot_numbers+=len(param['ellipse_centers'])
        plt.figure(1, figsize=(20,30))        

    lim_ux=max(time)+2
    if os.path.exists('energies_time'):
        sites = np.genfromtxt('energies_time')
        time_energy=np.arange(0,len(sites[:,0]), 1)*0.25
    start = np.argmax(ave(bulk, 10)<np.max(gau)*0.05)
    print('Equilibration point where bulk is <5% the maximum gaussian height is: '+str(np.round(ave(time_bulk, 10)[start], 2)))
    plt.figure(1, figsize=(20,30))

    plt.subplot(plot_numbers,1,1)
    plt.title('Raw gaussian height' ,  fontproperties=font1, fontsize=40,y=1.05)
    # plt.axvline(ave(time_bulk, 10)[start],ymin=0, ymax=0.70, linewidth=8,color='k')
    plt.plot(time,gau, color='blue',linewidth=4)
    plt.plot(ave(time, 10), ave(gau, 10), color='red',linewidth=4)
    plt.yticks(np.arange(0, 1.3,0.4), fontproperties=font1,  fontsize=30)#
    plt.xticks(np.arange(0, lim_ux,param['converge_x_interval']), fontproperties=font1,  fontsize=30)#
    plt.ylim(-0.1,1.2);plt.xlim(-2, lim_ux)
    plt.tick_params(axis='both', which='major', width=3, length=5, labelsize=35, direction='in', pad=10, right=False, top=False,labelbottom=False)
    plt.ylabel('Hills height \n(kJ mol$^{-1}$)', fontproperties=font2,fontsize=param['converge_labels'])

    plt.subplot(plot_numbers,1,2)
    plt.title('Bulk gaussian height' ,  fontproperties=font1, fontsize=40,y=param['converge_title_height'])
    plt.axvline(ave(time_bulk, 10)[start],ymin=0, ymax=0.70, linewidth=8,color='k')
    plt.plot(time_bulk ,bulk, color='blue',linewidth=4)
    plt.plot(ave(time_bulk, 10), ave(bulk, 10), color='red',linewidth=4)
    plt.yticks(np.arange(0, 1.3,0.4), fontproperties=font1,  fontsize=30)#
    plt.xticks(np.arange(0, lim_ux,param['converge_x_interval']), fontproperties=font1,  fontsize=30)#
    plt.ylim(-0.1,1.2);plt.xlim(-2, lim_ux)
    
    plt.ylabel('Hills height \n(kJ mol$^{-1}$)', fontproperties=font2,fontsize=param['converge_labels']) 
    if plot_numbers == 2:
        plt.xlabel('Time ($\mu$s)', fontproperties=font2,fontsize=30)
        plt.tick_params(axis='both', which='major', width=3, length=15, labelsize=35, direction='in', pad=10, right=False, top=False)
    else:
        plt.tick_params(axis='both', which='major', width=3, length=5, labelsize=35, direction='in', pad=10, right=False, top=False,labelbottom=False)
    offset=0
    if param['circle_plot']:
        for val, fig_num in enumerate(range(3,len(param['circle_centers'])+3,1)):
            center=np.where(np.sqrt(((param['circle_centers'][val][0]-dx)**2)+((param['circle_centers'][val][1]-dy)**2)) <= param['circle_area'][val])
            plot_converge_sites(plot_numbers, fig_num, time_bulk, start, time, gau, center, param, lim_ux)
            offset+=1
    if param['ellipse_plot']:
        for val, fig_num in enumerate(range(3+offset,len(param['ellipse_centers'])+3+offset,1)):
            center=ellipse_check_point(dx, dy, param['ellipse_centers'][val], param['ellipse_width'][val], param['ellipse_height'][val], param['ellipse_angle'][val])
            plot_converge_sites(plot_numbers, fig_num, time_bulk, start, time, gau, center, param, lim_ux)

    plt.xlabel('Time ($\mu$s)', fontproperties=font2,fontsize=param['converge_labels'])
    if plot_numbers == 2:
        plt.subplots_adjust( top=0.92, bottom=0.12, left=0.12,right=0.97, wspace=0.4, hspace=0.12)
    else:
        plt.subplots_adjust( top=0.955, bottom=0.075, left=0.12,right=0.97, wspace=0.4, hspace=0.08)
    plt.savefig('convergence.png', dpi=300)
    if show:
        plt.show()

def plot_converge_sites(plot_numbers, fig_num, time_bulk, start, time, gau, center, param, lim_ux):
    plt.subplot(plot_numbers,1,fig_num)
    plt.axvline(ave(time_bulk, 10)[start],ymin=0, ymax=0.70, linewidth=8,color='k')
    plt.title('Site '+str(fig_num-2) ,  fontproperties=font1, fontsize=40,y=param['converge_title_height'])#+' gaussian height'
    plt.scatter(time[center],gau[center], s=100, color='blue')
    plt.scatter(ave(time[center], 10), ave(gau[center], 10), s=25, alpha=0.3, color='red')
    plt.yticks(np.arange(0, 1.21,0.4), fontproperties=font1,  fontsize=30)
    plt.xticks(np.arange(0, lim_ux,param['converge_x_interval']), fontproperties=font1,  fontsize=30)#
    plt.ylim(-0.1,1.2);plt.xlim(-2, lim_ux)
    plt.ylabel('Hills height \n(kJ mol$^{-1}$)', fontproperties=font2,fontsize=param['converge_labels']) 
    if fig_num != plot_numbers:
        plt.tick_params(axis='both', which='major', width=3, length=15, labelsize=35, direction='in', pad=10, right=False, top=False,labelbottom=False)
    else:
        plt.tick_params(axis='both', which='major', width=3, length=15, labelsize=35, direction='in', pad=10, right=False, top=False)


def set_to_zero(energy):
    if energy[-150] < 0:
        # print(energy[-20:-1])
        energy = energy+(0-energy[-150])
        # print(energy[-20:-1])

    else:
        # print(energy[-20:-1])
        energy = energy-energy[-150]    
        # print(energy[-20:-1])
    return energy

def find_min(floatz):
    minz=np.round(floatz,-1)
    if minz>=floatz:
        minz-=5
    return minz

def write_pdb(pdb_file, z, dict):
    pdbline = "ATOM  %5d %4s %4s%1s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f"
    box_line="CRYST1 %8.3f %8.3f %8.3f  90.00  90.00  90.00 P 1           1\n"
    with open(pdb_file, 'w') as pdb:
        at_count=0
        for x in dict:
            for y in dict[x]:
                at_count+=1
                pdb.write(pdbline%((at_count, 'dg', 'head',' ',1,float(x), float(y), z, 1, dict[x][y]))+'\n')

def simple_fes(fes, param):
    dict0 = {}
    for position in fes:
        if position[2] < 5:
            x = (position[0]*-1+param['pdb_offset'][0])*10
            y = (position[1]*-1+param['pdb_offset'][1])*10
            x = np.round(x * 2) / 2
            y = np.round(y * 2) / 2
            if str(x) not in dict0:
                dict0[str(x)]={str(y):[]}
            if str(y) not in dict0[str(x)]:
                dict0[str(x)][str(y)] = []
            dict0[str(x)][str(y)].append(position[2])
    return dict0

def trim_fes(x_list,y_list, z_list, param):
    fes_complete = np.stack((x_list,y_list, z_list), axis=-1)
    fes_cut_x = fes_complete[np.logical_and(param['plot_trim'][0] < fes_complete[:,0],fes_complete[:,0] < param['plot_trim'][1])]
    fes_cut_y = fes_cut_x[np.logical_and(param['plot_trim'][0] < fes_cut_x[:,1],fes_cut_x[:,1] < param['plot_trim'][1])]
    return fes_cut_y 

def average_simple_fes(sim_fes):
    dict_mean = {}
    for x in sim_fes:
        for y in sim_fes[x]:
            if x not in dict_mean:
                dict_mean[x] = {}
            if len(sim_fes[x][y]) > 0:
                dict_mean[x][y] = np.mean(sim_fes[x][y]) 
    return dict_mean

def plot_pdb(param):
    print('reading in :'+param['fes'])
    x_list,y_list,z_list,e_list=readfes(param['fes'], param)    ### reads in energy landscape
    print('finding bulk')
    z_list, bulk, sorted_X_s, sorted_Y_s, sorted_X_l, sorted_Y_l = get_bulk(x_list,y_list,z_list, param)
    print('triming fes')
    fes_cut = trim_fes(x_list,y_list, z_list, param)
    print('simple fes out')
    simple_fes_dict=simple_fes(fes_cut, param)    ### reads in energy landscape
    mean_simple_fes_dict = average_simple_fes(simple_fes_dict)
    write_pdb(param['pdb_output'], param['pdb_offset'][2]*10,  mean_simple_fes_dict)

def reshape_fes(x,y,z,e):
    ###  reshapes x & y to be plotted in 2D 
    count=0
    ini = 0
    
    for i in range(len(x)):  ### gets length of X
        if y[i] == ini:
            count+=1 
        else:
            if 'check' not in locals():
                check = []
            else:
                check.append(count)
            ini = y[i]
            count = 1
    check.append(count)
    if check.count(check[0]) != len(check):
        print('you are missing values in the fes file')
        for i_val, i in enumerate(check):
            if i != check[0]:
                print(i_val+1, len(check), check[0], i)
        sys.exit()


    Z = z.reshape(int(len(x)/count),count)
    E = e.reshape(int(len(x)/count),count)
    X = np.arange(np.min(x), np.max(x),  (np.sqrt((np.min(x)- np.max(x))**2))/count)# 0.00353453608)#
    Y = np.arange(np.amin(y), np.amax(y), (np.sqrt((np.min(y)- np.max(y))**2))/(len(x)/count))
    X, Y = np.meshgrid(X, Y)

    return X,Y,Z,E

def get_bulk(x,y,z, param):
#### fetch bulk area    
    if os.path.exists(param['bulk_values']):  ### reads bulk value file to get bulk area
        coord=np.genfromtxt(param['bulk_values'], autostrip=True)
        sorted_X_s, sorted_Y_s, sorted_X_l, sorted_Y_l=coord[:,0],coord[:,1],coord[:,2],coord[:,3]
        shuffledxy = np.stack((x,y), axis=-1)
        bulk, sorted_X_s, sorted_Y_s, sorted_X_l, sorted_Y_l = find_bulk(x,y,z,shuffledxy,sorted_X_s, sorted_Y_s, sorted_X_l, sorted_Y_l)
    else:   ### finds bulk area from scratch (much slower)
        print('finding bulk area')
        bulk, sorted_X_s, sorted_Y_s, sorted_X_l, sorted_Y_l = bulk_val(x,y,z, param)

    z[ z==0 ] = np.ma.masked    ### removes unsampled area by converting to nan
    z=np.clip(z-bulk,np.nanmin(z)-bulk,param['plot_energy_max'])    
    z[ z>=param['plot_energy_max'] ] = np.ma.masked
    return z, bulk, sorted_X_s, sorted_Y_s, sorted_X_l, sorted_Y_l


def final_frame(param, error, show):
### intialisation
    start_plot=time.time()
    if not error:
        fig1 = plt.figure(1, figsize=(35,20))
        ax1 = fig1.add_subplot(111)#, aspect='equal')
    ### add picture
        if param['picture']:   ### background picture
            im1 = plt.imread(param['picture_file'])
            implot1 = plt.imshow(im1,aspect='equal', extent=(param['picture_loc'][0],param['picture_loc'][1],param['picture_loc'][2],param['picture_loc'][3]), alpha=1)
    ### read in fes
        x,y,z,e=readfes(param['fes'], param)    ### reads in energy landscape
        z, bulk, sorted_X_s, sorted_Y_s, sorted_X_l, sorted_Y_l = get_bulk(x,y,z, param)
        X,Y,Z,E = reshape_fes(x,y,z,e)

        ### allows manually defined maxz and minz
        maxz=np.nanmax(Z) if 'plot_energy_max' not in param else param['plot_energy_max']
        minz=find_min(np.nanmin(Z)) if 'plot_energy_min' not in param else param['plot_energy_min']
        contour = np.array(np.arange(minz, maxz+0.1, param['plot_step']))
    ### FES shading and contours
        cax = ax1.contourf(X,Y,Z,np.arange(np.round(minz,0), param['plot_energy_max']+0.1, param['plot_step']), cmap=plt.get_cmap('coolwarm_r'),norm=MidpointNormalize(midpoint=0.), alpha=0.5, vmin = minz, vmax = maxz )# , alpha=0.25)
        con = ax1.contour(X,Y,Z,contour, linewidths=4, cmap=plt.get_cmap('coolwarm_r'), linestyles='-',norm=MidpointNormalize(midpoint=0.), alpha=1, vmin = minz, vmax = maxz )#, alpha=1)#colors='k'# cax = ax1.contourf(X,Y,Z,np.arange(-100, 15, 5), cmap=cm.coolwarm, vmin = minz, vmax = 15 , alpha=1)

    ### colourbar   np.arange(np.round(minz,-1), maxz+1, step*2))
        cbar = fig1.colorbar(cax, ticks=np.arange(np.round(minz,0), param['plot_energy_max']+0.1, param['plot_colour_bar_tick']))#.set_alpha(1)
        ticklabs = cbar.ax.get_yticklabels()
        cbar.ax.set_yticklabels(ticklabs, ha='right', va='center', fontsize=60)
        cbar.set_label('Free energy (kJ mol$^{-1}$)',fontsize=60, labelpad=40)
        cbar.ax.yaxis.set_tick_params(pad=200)
        cbar.set_alpha(1)
        cbar.draw_all()
        text_params = {'ha': 'center', 'va': 'center', 'family': 'sans-serif','fontweight': 'bold'}

    ### FES highlights circle or ellipse
        if param['circle_plot']:
            for val, ring in enumerate(param['circle_centers']):
                circle = plt.Circle((ring[0],ring[1]), param['circle_area'][val],linewidth=10,edgecolor='k',facecolor='none', zorder=2)
                ax1.add_artist(circle)
        if param['ellipse_plot']:
            for val, ring in enumerate(param['ellipse_centers']):
                ellipse = patches.Ellipse(xy=(ring[0],ring[1]), width=param['ellipse_width'][val], height=param['ellipse_height'][val], angle=param['ellipse_angle'][val],linewidth=10,edgecolor='k',facecolor='none', zorder=2) # original
                ax1.add_artist(ellipse)
    ### Tick parameters
        ax1.set_xlabel(param['CV1'], fontproperties=font1,  fontsize=80);ax1.set_ylabel(param['CV2'], fontproperties=font1,  fontsize=80)
        plt.xticks(np.arange(-180, 180.1,param['plot_interval_x']), fontproperties=font1,  fontsize=80)
        plt.yticks(np.arange(-180, 180.1,param['plot_interval_y']), fontproperties=font1,  fontsize=80)#
        ax1.tick_params(axis='both', which='major', width=3, length=5, labelsize=80, direction='out', pad=10)
    ### plot bulk area in dotted lines
        if param['bulk_outline']:
            plt.plot(sorted_X_l, sorted_Y_l, color='k', linestyle='--', linewidth=6)
            plt.plot(sorted_X_s, sorted_Y_s, color='k', linestyle='--', linewidth=6)
    ### 1D stuff bars
        if param['1d']:
            d_landscape, x_landscape, y_landscape, z_landscape =[],[],[],[]
            for line in range(len(param['start_points'])):
                distance,d1x,d1y,d1z,point_lower_x,point_lower_y,point_upper_x,point_upper_y=strip_1d(x,y,z, param['start_points'][line], param['end_points'][line], param['search_width'][line])
                plt.plot([point_lower_x[0],point_lower_x[-1]],[point_lower_y[0],point_lower_y[-1]], color='red',linewidth=10, zorder=2)
                plt.plot([point_upper_x[0],point_upper_x[-1]],[point_upper_y[0],point_upper_y[-1]], color='red',linewidth=10, zorder=2)
                plt.plot(d1x,d1y, color='k',linewidth=10, zorder=2)
                d_landscape.append(distance)
                x_landscape.append(d1x)
                y_landscape.append(d1y)
                z_landscape.append(d1z)
                # if save_plot != None:
                write_file(distance,d1x,d1y,d1z, line+1)
    ### plot limits
        minx, maxx=min(x)-0.1 ,max(x)+0.1
        miny, maxy=min(y)-0.1,max(y)+0.1
        if param['plot_trim'].any():
            minx, maxx, miny, maxy = param['plot_trim'][0], param['plot_trim'][1], param['plot_trim'][2], param['plot_trim'][3]
        plt.xlim(minx,maxx);plt.ylim(miny,maxy)
        plt.subplots_adjust(top=0.972, bottom=0.13,left=0.168,right=0.90, hspace=0.2, wspace=0.2)
        plt.savefig(param['fes'].replace('.dat','')+'.png', dpi=300)
        if param['1d']:
            fig2 = plt.figure(2, figsize=(35,20))### error plot   [0,2.5,5,7.5,10,12.5,15]
            for val, landscape in enumerate(d_landscape):
                plt.plot(landscape, z_landscape[val], linewidth=5, label='site '+str(val+1))
            plt.tick_params(axis='both', which='major', width=3, length=5, labelsize=15, direction='in', pad=10, right=False, top=False)
            plt.xlabel('Distance (nm)', fontproperties=font2,fontsize=15);plt.ylabel('Energy (kJ mol$^{-1}$)', fontproperties=font2,fontsize=15) 
            plt.legend(prop={'size': 10}, bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=len(d_landscape), mode="expand", borderaxespad=0.)
            # plt.savefig(fes+'_1D.png', dpi=300)
        if param['1d']:
            plt.savefig(param['fes'].replace('.dat','')+'_1D.png', dpi=300)
        else:
            plt.savefig(param['fes'].replace('.dat','')+'.png', dpi=300)
        if show:
            plt.show()
    else:
#### Error analysis plot
        x,y,z,e=readfes(param['fes'], param)
    ###  reshapes x & y to be plotted in 2D 
        count=0
        for i in range(len(x)):  ### gets length of X
            if y[i]== y[0]:
                count+=1
            else:
                break
        Z = z.reshape(int(len(x)/count),count)
        E = e.reshape(int(len(x)/count),count)
        X = np.arange(np.min(x), np.max(x),  (np.sqrt((np.min(x)- np.max(x))**2))/count)# 0.00353453608)#
        Y = np.arange(np.amin(y), np.amax(y), (np.sqrt((np.min(y)- np.max(y))**2))/(len(x)/count))
        X, Y = np.meshgrid(X, Y)
        fig1 = plt.figure(1, figsize=(35,20))### 
        step = 2.5
        ax = fig1.add_subplot(111, aspect='equal')
    ### FES highlights circle or ellipse
        if param['circle_plot']:
            for val, ring in enumerate(param['circle_centers']):
                circle = plt.Circle((ring[0],ring[1]), param['circle_area'][val],linewidth=10,edgecolor='k',facecolor='none', zorder=2)
                ax1.add_artist(circle)
        if param['ellipse_plot']:
            for val, ring in enumerate(param['ellipse_centers']):
                ellipse = patches.Ellipse(xy=(ring[0],ring[1]), width=param['ellipse_width'][val], height=param['ellipse_height'][val], angle=param['ellipse_angle'][val],linewidth=10,edgecolor='k',facecolor='none', zorder=2) # original
                ax1.add_artist(ellipse)
        cax = ax.contourf(X,Y,E, np.arange(0,np.nanmax(e)+step,step),cmap=plt.get_cmap('coolwarm'),norm=MidpointNormalize(midpoint=0.), alpha=0.25, vmin = np.nanmin(e), vmax = np.nanmax(e))# , alpha=0.25)
        cone = ax.contour(X,Y,E, np.arange(0,np.nanmax(e)+step,step),linewidths=8, cmap=plt.get_cmap('coolwarm'), linestyles='-',norm=MidpointNormalize(midpoint=0.), alpha=1, vmin = np.nanmin(e), vmax = np.nanmax(e) )#, alpha=1)#colors='k'# cax = ax1.contourf(X,Y,Z,np.arange(-100, 15, 5), cmap=cm.coolwarm, vmin = minz, vmax = 15 , alpha=1)
        # conz = ax.contour(X,Y,Z,[5], linewidths=8, cmap=plt.get_cmap('coolwarm_r'), linestyles='-',norm=MidpointNormalize(midpoint=0.), alpha=1, vmin = minz, vmax = maxz )#, alpha=1)#colors='k'# cax = ax1.contourf(X,Y,Z,np.arange(-100, 15, 5), cmap=cm.coolwarm, vmin = minz, vmax = 15 , alpha=1)

    #### colourbar   np.arange(np.round(minz,-1), maxz+1, step*2))
        cbar = fig1.colorbar(cax, ticks=np.arange(0,np.nanmax(e)+step,step))#np.append(np.arange(np.round(minz,0), -2.5, step*2),5))#.set_alpha(1)
        ticklabs = cbar.ax.get_yticklabels()
        cbar.ax.set_yticklabels(ticklabs, ha='right', va='center', fontsize=60)
        cbar.set_label('Free energy (kJ mol$^{-1}$)',fontsize=60, labelpad=40)
        cbar.ax.yaxis.set_tick_params(pad=200)
        cbar.set_alpha(1)
        cbar.draw_all()
        # print(np.nanmin(E), np.nanmax(E))

        plt.xlim(min(x)-0.1,max(x)+0.1);plt.ylim(min(y)-0.1,max(y)+0.1)
        plt.subplots_adjust(top=0.972, bottom=0.13,left=0.168,right=0.90, hspace=0.2, wspace=0.2)
        
        # plt.savefig(param['fes'].replace('.dat','')+'_error.png', dpi=300)
        if show:
            plt.show()
    #####
        # plt.figure(3, figsize=(20,10))  

        # for val,energy in enumerate(landscape):
        #   energy = set_to_zero(energy)
        #   en_val_prev=20
        #   up=0
        #   for index, en_val in enumerate(energy):
        #       if en_val <= en_val_prev:
        #           en_val_prev=en_val
        #           up=0
        #       if up==10:
        #           zmin=np.sqrt(((x_landscape[val][index]-x_landscape[val][0])**2)+((y_landscape[val][index]-y_landscape[val][0])**2))
        #           break
        #       else:
        #           up+=1
        #   index =np.where(np.min(energy)==energy)
        #   zmin=np.sqrt(((x_landscape[val][index]-x_landscape[val][0])**2)+((y_landscape[val][index]-y_landscape[val][0])**2))
        #   distance=np.sort(np.sqrt(((x_landscape[val]-x_landscape[val][0])**2)+((y_landscape[val]-y_landscape[val][0])**2)))-np.mean(zmin)
        #   write_file(distance,energy, val+1)
        #   plt.plot(distance, energy,linewidth=4, label='Site: '+str(val+1))
        #   print(val+1, np.min(energy))


def strip(param, fes):
    if os.path.exists(param['prefix']+str(fes)+'.dat'):
        x,y,z, e=readfes(param['prefix']+str(fes)+'.dat', param)
        start_time = time.time()
        bulk, sorted_X_s, sorted_Y_s, sorted_X_l, sorted_Y_l = bulk_val(x,y,z, param)
        output=[fes,bulk]
        if param['circle_plot']:
            if len(param['circle_centers']) > 0:
                for val, ring in enumerate(param['circle_centers']):
                    minima=np.where(np.sqrt(((ring[0]-x)**2)+((ring[1]-y)**2)) <= param['circle_area'][val])
                    output.append(np.round(min(z[minima])-bulk,4))
        if param['ellipse_plot']:
            if len(param['ellipse_centers']) > 0:
                for val, ring in enumerate(param['ellipse_centers']):
                    minima=ellipse_check_point(x, y, ring, param['ellipse_width'][val], param['ellipse_height'][val], param['ellipse_angle'][val])
                    output.append(np.round(min(z[minima])-bulk,4))
        if len(output) > 0:
            return output
        else:
            sys.exit('Ellipse or circle points have not been selected')

def ellipse_check_point(x, y, center, width, height, angle):
    cos_angle = np.cos(math.pi-np.radians(angle))
    sin_angle = np.sin(math.pi-np.radians(angle))
    xc = x - center[0]
    yc = y - center[1]
    xct = xc * cos_angle - yc * sin_angle
    yct = xc * sin_angle + yc * cos_angle 
    rad_cc = (xct**2/(width/2.)**2) + (yct**2/(height/2.)**2)
    colors_array = []
    minima = np.where(rad_cc < 1)
    return minima

@jit(nopython=True)
def bootstrap(z):
    if not np.isnan(z).all():
        bs_sample = np.random.choice(z, size=10000)
        return [np.mean(bs_sample), np.std(bs_sample)]
    return [np.mean(z), np.std(z)]

def average_fes(param, fes, sorted_X_s, sorted_Y_s, sorted_X_l, sorted_Y_l):
    print('reading in '+param['prefix']+str(fes)+'.dat')
    start_time = time.time()
    x,y,z, e=readfes(param['prefix']+str(fes)+'.dat', param)
    shuffledxy = np.stack((x,y), axis=-1)
    bulk, sorted_X_s, sorted_Y_s, sorted_X_l, sorted_Y_l = find_bulk(x,y,z,shuffledxy,sorted_X_s, sorted_Y_s, sorted_X_l, sorted_Y_l)
    z[ z==0 ] = np.nan
    z=np.clip(z-bulk,np.nanmin(z)-bulk,20)
    return x,y,z

def get_frames(param):
    # param = parameters()
    os.system('mkdir analysis')
    os.chdir('analysis')
    xtc_files=[]
    for filename in os.listdir('../'):
        if filename.endswith(".xtc"):
            # xtc = re.search('(*.)\.', filename)[0]
            # print(xtc)
            # xtc = re.search('\.(.*)\.', filename)[1]
            xtc_files.append(filename[:-4])
    pool = mp.Pool(mp.cpu_count())

    # test = pool.map_async(gromacs, ['gmx distance -f ../*'+xtc+'.xtc -s ../*tpr -n ../../build/index.ndx -oxyz '+xtc+'.xvg -select \'cog of group \"trans\" plus cog of group \"pip2_head\"\'' for xtc in xtc_files]).get()
    # pool.join
    
    for val, xtc in enumerate(xtc_files):
        file_out=np.genfromtxt('../'+xtc+'.xvg', autostrip=True, comments='#!',skip_header=1)
        for ring_num, ring in enumerate(param['circle_centers']):
            loc=np.where(np.sqrt(((float(ring[0])-file_out[:,1])**2)+((float(ring[1])-file_out[:,2])**2)) <= param['circle_area'][ring_num])
            time_loc=file_out[:,0][loc]*200000
            start_time = time.time()

            test = pool.map_async(gromacs, ['echo 0 | gmx trjconv -pbc res -f ../*'+xtc+'.xtc -s ../*tpr -b '+str(time_stamp)+' -e '+str(time_stamp)+' -o ring_'+str(ring_num+1)+'_'+xtc+'_'+str(int(time_stamp))+'_ind.xtc' for time_stamp in time_loc]).get()
            pool.join

            print(time.time()-start_time)
            out=gromacs('gmx trjcat -f ring_'+str(ring_num+1)+'_'+xtc_files[val]+'_*.xtc -o all_ring_'+str(ring_num+1)+'_'+xtc_files[val]+'.xtc' )

    os.system('rm r*_part*')
    for ring, ring_loc in enumerate(param['circle_centers']):
        out=gromacs('gmx trjcat -f all_ring_'+str(ring+1)+'_*.xtc -o all_ring_'+str(ring+1)+'.xtc' )
    os.system('rm *_part*')
    os.chdir('../..')


###############  1D bits
def dy(distance, m):
    return m*dx(distance, m)
def dx(distance, m):
    return np.sqrt(distance/(m**2+1))
 

def points_1d(x,y,z,point_lower_x,point_lower_y,point_upper_x,point_upper_y,shuffledxy, val, line_x, line_y):

        x_high= np.max([point_upper_x[val],point_upper_x[val+1],point_lower_x[val],point_lower_x[val+1]])
        x_low = np.min([point_upper_x[val],point_upper_x[val+1],point_lower_x[val],point_lower_x[val+1]])
        y_high= np.max([point_upper_y[val],point_upper_y[val+1],point_lower_y[val],point_lower_y[val+1]])
        y_low = np.min([point_upper_y[val],point_upper_y[val+1],point_lower_y[val],point_lower_y[val+1]])
        p = path.Path([(point_upper_x[val], point_upper_y[val]), (point_upper_x[val+1], point_upper_y[val+1]), \
                (point_lower_x[val+1],point_lower_y[val+1]), (point_lower_x[val],point_lower_y[val])])
        square=np.where( np.logical_and(np.logical_and(x_low < x, x < x_high),np.logical_and(y_low < y, y < y_high)))
        bulk_values =  p.contains_points(shuffledxy[square])
        try:
            if len(z[square][bulk_values]) > 0 and not np.all(np.isnan(x)):
                if np.nanmin(z[square][bulk_values]) < 20:
                    zmin_index=int(np.argwhere(z[square][bulk_values]==np.nanmin(z[square][bulk_values])))
                    xcoord, ycoord, zcoord = shuffledxy[square][bulk_values][zmin_index][0], shuffledxy[square][bulk_values][zmin_index][1], round(float(np.nanmin(z[square][bulk_values])),2) 
                    return xcoord,ycoord,zcoord, line_x[val], line_y[val]
        except:
            return None,None,None,None,None


def strip_1d(x,y,z, start_points, end_points, distance):
    length_CV=np.sqrt((start_points[0]-end_points[0])**2+(start_points[1]-end_points[1])**2)
    number_points=length_CV/0.005
    CV=np.linspace(0,int(length_CV),int(number_points),endpoint=False)
    point_lower_x, point_lower_y,point_upper_x,point_upper_y=np.array([]),np.array([]),np.array([]),np.array([])

    if start_points[0]!=end_points[0] and start_points[1]!=end_points[1]:
        slope, intercept, r_value, p_value, std_err = stats.linregress([start_points[0], end_points[0]],[start_points[1], end_points[1]])
        perp_slope=(-1/slope)
        perp_intercept=(perp_slope*start_points[0]-start_points[1])*-1
        if end_points[0] < start_points[0]:
            line_x = np.array(start_points[0]-CV*np.sqrt(1/(1+slope**2)))
            line_y = np.array(start_points[1]-CV*slope*np.sqrt(1/(1+slope**2)))
        else:
            line_x = np.array(start_points[0]+CV*np.sqrt(1/(1+slope**2)))
            line_y = np.array(start_points[1]+CV*slope*np.sqrt(1/(1+slope**2)))
        for val, x_val in enumerate(line_x):
            y_val=line_y[val]
            perp_intercept=(perp_slope*x-y)*-1
            point_lower_x= np.append(point_lower_x, x_val+dx(distance,perp_slope))
            point_lower_y= np.append(point_lower_y, y_val+dy(distance,perp_slope))
            point_upper_x= np.append(point_upper_x, x_val-dx(distance,perp_slope))
            point_upper_y= np.append(point_upper_y, y_val-dy(distance,perp_slope))
    # return point_lower_x*invert,point_lower_y,point_upper_x*invert,point_upper_y
    else:
        if start_points[0]==end_points[0]:
            line_x = np.array(start_points[0]+np.zeros(len(CV)))
            line_y = np.array(start_points[1]+CV)
            for val, y_val in enumerate(CV):
                point_lower_x= np.append(point_lower_x, start_points[0]-distance)
                point_lower_y= np.append(point_lower_y, start_points[1]+y_val)
                point_upper_x= np.append(point_upper_x, start_points[0]+distance)
                point_upper_y= np.append(point_upper_y, start_points[1]+y_val)
        if start_points[1]==end_points[1]:
            line_x = np.array(start_points[0]+CV)
            line_y = np.array(start_points[1]+np.zeros(len(CV)))
            for val, x_val in enumerate(CV):
                point_lower_x= np.append(point_lower_x, start_points[0]+x_val)
                point_lower_y= np.append(point_lower_y, start_points[1]-distance)
                point_upper_x= np.append(point_upper_x, start_points[0]+x_val)
                point_upper_y= np.append(point_upper_y, start_points[1]+distance)               
    shuffledxy = np.stack((x,y), axis=-1)
    pool = mp.Pool(mp.cpu_count())
    xyz = pool.starmap_async(points_1d, [(x,y,z,point_lower_x,point_lower_y,point_upper_x,point_upper_y,shuffledxy,frame, line_x, line_y) for frame in range(len(line_x)-1)]).get()
    pool.close
    xyz = np.array([col for col in xyz if col != None])
    xyz = np.delete(xyz,np.where(xyz[:,0]==None), 0).astype(np.float64)
    return np.sqrt(((xyz[:,3]-float(line_x[0]))**2)+((xyz[:,4]-float(line_y[0]))**2)), xyz[:,0], xyz[:,1], xyz[:,2],point_lower_x,point_lower_y,point_upper_x,point_upper_y


def average_1d(param, fes_frame):   ### not tested
    start_time = time.time()
    x,y,z,e=readfes(param['prefix']+str(fes_frame)+'.dat', param)
    try:  ### reads bulk value file to get bulk area
        coord=np.genfromtxt('bulk_values', autostrip=True)
        sorted_X_s, sorted_Y_s, sorted_X_l, sorted_Y_l=coord[:,0],coord[:,1],coord[:,2],coord[:,3]
        shuffledxy = np.stack((x,y), axis=-1)
        bulk, sorted_X_s, sorted_Y_s, sorted_X_l, sorted_Y_l = find_bulk(x,y,z,shuffledxy,sorted_X_s, sorted_Y_s, sorted_X_l, sorted_Y_l)
    except:   ### finds bulk area from scratch (much slower)
        print('finding bulk area')
        bulk, sorted_X_s, sorted_Y_s, sorted_X_l, sorted_Y_l = bulk_val(x,y,z, param)
        z=np.clip(z-bulk,np.nanmin(z)-bulk,20)
    for line in range(len(param['start_points'])):
        if not os.path.exists(param['1d_location']+'/1D_landscape_site-'+str(line+1)+'_'+str(fes_frame)):
            distance,d1x,d1y,d1z,point_lower_x,point_lower_y,point_upper_x,point_upper_y=strip_1d(x,y,z, param['start_points'][line], param['end_points'][line], param['search_width'][line])
            with open(param['1d_location']+'/1D_landscape_site-'+str(line+1)+'_'+str(fes_frame), 'w') as landscape:
                for line in range(len(distance)):
                    landscape.write(str(np.round(distance[line],3))+'\t'+str(d1z[line])+'\t'+str(np.round(d1x[line],5))+'\t'+str(np.round(d1y[line],5))+'\n')
            landscape.close()
    # print(fes_frame, np.round(time.time()-start_time, 2))

def bootstrap_1d(param, show):
    for site in range(1,len(param['start_points'])+1):
        init=True
        energy=[]
        files_range=[]
        for root, dirs, files in os.walk(param['1d_location']):
            for filename in files:
                if '1D_landscape_site-'+str(site)+'_' in filename :
                    files_range.append(filename)
            break
        for fes_frame in files_range:
            if int(fes_frame.split('_')[3]) > param['equilibration']/param['stride']:
                data=np.genfromtxt(param['1d_location']+'/'+fes_frame)

                bulk=np.where(np.logical_and(data[:,0]>param['ref_min'][site-1], data[:,0]<param['ref_max'][site-1]))
                bulk_val=(np.mean(data[:,1][bulk]))
                data[:,1]=data[:,1]-bulk_val
                if init:
                    init=False
                    landscapes = {str(np.round(CV,3)): [] for CV in np.linspace(0,np.max(data[:,0])+0.05,(np.max(data[:,0])+0.05)*200,endpoint=False)}
                for val,CV in enumerate(data[:,0]):
                    landscapes[str(CV)].append(data[:,1][val])
            else:
                pass
        for CV in np.linspace(0,7,1400,endpoint=False):
            if str(np.round(CV,3)) in landscapes:
                if len(landscapes[str(np.round(CV,3))]) >1 and CV < param['cv_max'][site-1] and CV > param['cv_min'][site-1]:
                    bs_sample = np.random.choice(landscapes[str(np.round(CV,3))], size=10000)
                    energy.append([np.round(CV,3), np.mean(bs_sample), np.std(bs_sample)])
        energy=np.array(energy)
        plt.figure(1, figsize=(20,20))
        plt.subplot(len(param['start_points']),1,site)      
        cutoff_new = float(param['ref_max'][site-1]-energy[:,0][np.where(energy[:,1]==np.min(energy[:,1]))])
        ref_min_new = float(param['ref_min'][site-1]-energy[:,0][np.where(energy[:,1]==np.min(energy[:,1]))])
        energy[:,0] = energy[:,0] - energy[:,0][np.where(energy[:,1]==np.min(energy[:,1]))]
        cutoff_max = energy[:,0][np.where(energy[:,1]==np.min(energy[:,1]))]

        if cutoff_max < np.max(energy[:,0]):
            cutoff_max=np.max(energy[:,0])
        plt.title('Site '+str(site) ,  fontproperties=font1, fontsize=40,y=param['boot_title_height'])#+' gaussian height'

        plt.plot(energy[:,0], energy[:,1], linewidth=3, color='red')
        plt.fill_between(energy[:,0], energy[:,1]-energy[:,2], energy[:,1]+energy[:,2], alpha=0.3, facecolor='black')
        error=np.mean(energy[:,2][np.where(np.logical_and(energy[:,0]>ref_min_new,energy[:,0]<cutoff_new))])+energy[:,2][np.where(energy[:,1]==np.min(energy[:,1]))]
        plt.annotate(str(int(np.round(np.min(energy[:,1]),0)))+' $\pm$ '+str(int(np.round(error,0)))+' kJ mol$^{-1}$', xy=(cutoff_max-0.1, param['boot_label_y']), size =25, bbox=bbox, ha="right", va="top")
        plt.yticks(np.arange(-500, param['boot_energy_max']+1,param['boot_step_yaxis']), fontproperties=font1,  fontsize=35)
        plt.xticks(np.arange(-3, 10,0.5), fontproperties=font1,  fontsize=35)#
        if site<len(param['start_points']):
            plt.tick_params(axis='both', which='major', width=3, length=5, labelsize=35, direction='in', pad=10, right=False, top=False,labelbottom=False)
        else:
            plt.tick_params(axis='both', which='major', width=3, length=5, labelsize=35, direction='in', pad=10, right=False, top=False)
            plt.xlabel('CV (nm)', fontproperties=font2,fontsize=param['boot_labels_size']) 

        plt.ylabel('Energy \n(kJ mol$^{-1}$)', fontproperties=font2,fontsize=param['boot_labels_size']) 
        # plt.xlim(0,cutoff)
        plt.xlim(np.min(energy[:,0]),cutoff_max)
        plt.ylim(np.min(energy[:,1])*1.1,20)
    plt.savefig('energy_landscape_sites_bootstraped_1d.png', dpi=300)
    if show:
        plt.show()


def plot_CV(param, show):
    plt.close()
    plt.rcParams['mathtext.default'] = 'regular'
    plt.rcParams['axes.linewidth'] = 3
    hills_files=[]
    for walker in range(int(param['walker_range'][0]),int(param['walker_range'][1])+1):
        if os.path.exists('HILLS.'+str(walker)):
            hills_files.append('HILLS.'+str(walker))    
    hills_x, hills_y=hills_files.copy(), hills_files.copy()
    time, dx, dy, gau=[],[],[],[]
    count=0
    check=False
    fig1 = plt.figure(1, figsize=(17.9,20))

    pool = mp.Pool(mp.cpu_count())
    xy = pool.starmap(multi_read_hills, [(hill_val, hills_file)  for hill_val, hills_file in enumerate(hills_files)])
    pool.close
    x_min, x_max, y_min,y_max =0, 0, 0, 0
    for hills in xy:
        hills=np.array(hills)
        if x_min > np.min(hills[:,0]): 
            x_min = np.min(hills[:,0]) 
        if x_max < np.max(hills[:,0]): 
            x_max = np.max(hills[:,0]) 
        if y_min > np.min(hills[:,1]): 
            y_min = np.min(hills[:,1]) 
        if y_max < np.max(hills[:,1]): 
            y_max = np.max(hills[:,1]) 

    x_min, x_max, y_min, y_max = x_min*1.1, x_max*1.1, y_min*1.1,y_max*1.1
    for hill_val, hills_file in enumerate(xy):
        ax1 = fig1.add_subplot(5,4, hill_val+1, aspect='equal')
        plt.title(str(hill_val+1) ,  fontproperties=font1, fontsize=30,y=param['cv_title_height'])
        ran=np.linspace(0,1,len(xy[hill_val][hill_val:,0]))
        ax1.scatter(xy[hill_val][hill_val:,0], xy[hill_val][hill_val:,1],s=1,c = cm.coolwarm(ran))
        if param['circle_plot']: 
            for val, ring in enumerate(param['circle_centers']):
                circle = plt.Circle((ring[0],ring[1]), param['circle_area'][val],linewidth=2,edgecolor='k',facecolor='none', zorder=2)
                ax1.add_artist(circle)
        if param['ellipse_plot']:
            for val, ring in enumerate(param['ellipse_centers']):
                ellipse = patches.Ellipse(xy=(ring[0],ring[1]), width=param['ellipse_width'], height=param['ellipse_height'], angle=param['ellipse_angle'],linewidth=10,edgecolor='k',facecolor='none', zorder=2) 
                ax1.add_artist(ellipse)
        plt.yticks(np.arange(-10, 10,param['cv_x_interval']), fontproperties=font1)#
        plt.xticks(np.arange(-10, 10,param['cv_y_interval']), fontproperties=font1)#
        rows=[1,5,9,13,17]
        lef, bot=False,False    
        if hill_val+1 in rows:
            plt.ylabel(param['CV2'], fontproperties=font2,fontsize=param['cv_labels_size']) 
            lef=True
        if hill_val+1 in np.arange(len(hills_files), len(hills_files)-4,-1):
            plt.xlabel(param['CV1'], fontproperties=font2,fontsize=param['cv_labels_size'])
            bot=True  
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        ax1.tick_params(axis='both', which='major', width=3, length=5, labelsize=param['cv_tick_size'], direction='in', pad=10 , labelbottom=bot, labelleft=lef,right=True, top=True, left=True, bottom=True)
        plt.subplots_adjust(left=0.1, wspace=0.1, hspace=0.1, top=0.975, bottom=0.08)
    plt.savefig('hills_CV_trace.png', dpi=300)
    if show:
        plt.show()

def multi_read_hills(hill_val, hills_file):
    hills=[]
    with open(hills_file, 'r') as hills_file_in:
        for line in hills_file_in:
            if not line[0] in ['#', '@']:
                if np.round(np.round(float(line.split()[0]), 3),3).is_integer():
                    hills.append([float(line.split()[1]), float(line.split()[2])])
    hills = np.array(hills)
    if len(hills[:,0]) != len(hills[:,1]):
        sys.exit('error with HILLS file: '+hill_val)
    return hills


def timecourse(param, show):
    plt.close()
    min_energy=0
    count=0
    run_ylim=False
    plt.figure(1, figsize=(20,10))
    for site in range(1,len(param['start_points'])+1):
        count+=1
        if site==len(param['start_points']):
            run_ylim=True
        energy=[]
        files_range, number_range=[],[]
        for root, dirs, files in os.walk(param['1d_location']):
            for filename in files:
                if '1D_landscape_site-'+str(site)+'_' in filename :
                    files_range.append(filename)
                    number_range.append(int(filename.split('_')[3]))
            break
        number_range=np.sort(number_range)
        energy_min=np.array([])
        for f in number_range:
            data=np.genfromtxt(param['1d_location']+'/1D_landscape_site-'+str(site)+'_'+str(f), autostrip=True)
            try:
                bulk=np.where(np.logical_and(data[:,0]>param['cv_min'][site-1], data[:,0]<param['cv_max'][site-1]))
            except:
                pass
                # print(site, f)
                # print(len(cutoff))
                # print(cutoff[site-1]-cut, cutoff[site-1])

            bulk_val=(np.mean(data[:,1][bulk]))
            data_ori=data.copy()
            data[:,1]=data[:,1]-bulk_val
            energy_min=np.append(energy_min, np.nanmin(data[:,1]))
            if site==3 and  str(np.nanmin(data[:,1])) == 'nan':
                print(data_ori[:,1] ,data[:,1], bulk_val,  bulk)


        plt.plot(np.array(number_range)*0.25, energy_min, label='site: '+str(site), linewidth=4)

        plt.yticks(np.arange(-500, 101,param['step_yaxis']), fontproperties=font1,  fontsize=30)
        plt.xticks(np.arange(0, 300,param['step_xaxis']), fontproperties=font1,  fontsize=30)#
        # print(energy_min)
        if np.min(energy_min) < min_energy:
            min_energy = np.min(energy_min)
        if run_ylim:
            plt.ylim(np.min(min_energy)-5,5)
            plt.legend(prop={'size': 20}, bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=count, mode="expand", borderaxespad=0.)

        plt.xlim(-5, np.max(np.array(number_range)*0.25)+5)
        plt.ylabel('Energy \n(kJ mol$^{-1}$)', fontproperties=font2,fontsize=20) 
        plt.xlabel('Time ($\mu$s)', fontproperties=font2,fontsize=20)
        plt.tick_params(axis='both', which='major', width=3, length=5, labelsize=20, direction='in', pad=10, left=True, bottom=True)
        plt.subplots_adjust(left=0.16, wspace=0, hspace=0, top=0.9, bottom=0.11)

    plt.savefig('energy_timecourse.png', dpi=300)
    if show:
        plt.show()

def info():
    print('\nThis script will allow you to analyse 2D well tempered metadynamics simulations\n \
            It is contains several functions which are controlled by a input file.\n\
            This allows a reproducible workflow which can be released with the simulation raw data.\n\n')

    print('There are multiple functions available for analysing WTMetaD.\n')
    
    print('These are all controlled from within the input file specified by the \'-input\' flag.')
    print('Please see the README file for information on available variables\n')
    
    print('Using the \'-f\' flag you can select one of the following:\n')
    
    print('-f sort \t\t sorts the merged HILLS fill according to the deposition timestamp.')
    print('-f skip \t\t sorts the merged HILLS fill according to the deposition timestamp and writes out every 1 ns.')
    print('-f plot \t\t plots the 2D landscapes using the parameters in the input file.')
    print('-f concat \t\t concatonates FES landscapes and bootstraps each coordinate')
    print('-f strip \t\t strips the energy minima from the areas selected by circles or ellipses')
    print('-f converge \t\t plots the raw, bulk and selected areas from the HILLS file.')
    print('-f frames \t\t strips out all trajectory frames from within the selected areas.')
    print('-f 1d_time \t\t finds the minimum free energy path from a range of 2D FES landscapes.')
    print('-f boot \t\t Bootstraps the minimum free energy path landscapes')
    print('-f cv_plot \t\t plots the CVs coloured over time')
    print('-f timecourse \t\t plots the 1D energy minima over time.')
    sys.exit('\n')



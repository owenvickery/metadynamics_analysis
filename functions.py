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
        sys.exit('cannot find parameter file: '+args.input)
    return param

def check_variable(setting, param):
    for var in setting:
        if var not in param:
            sys.exit('The variable '+var+' is missing')

def split_setting(variable, setting):
    string_variables = ['fes', 'picture_file', 'bulk_values', 'CV1', 'CV2', 'prefix', '1d_location', 'HILLS_skipped']
    int_variables = ['start', 'end']
    float_variables = ['invert','minz','step','lim_ux','lim_ly','label_loc_x','label_loc_y','title_height','lab_y', 'labels_size', 'step_yaxis',
                       'interval_x', 'interval_y','energy_max', 'colour_bar_tick', 'search_width','cut', 'equilibration', 'stride']
    complex_variables = ['x_points', 'y_points', 'ring_location']
    list_variables = ['picture_loc', 'min_cv', 'cutoff', 'circle_area', 'ellipse_height', 'ellipse_width', 'ellipse_angle', 'cutoff', 
                      'min_cv', 'walker_range']
    T_F_variables = ['circle_plot','ellipse_plot', 'picture', '1d', 'bulk_outline']
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
        sys.exit('The variable '+variable+'is not correct')


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
    for line in open(files, 'r').readlines():
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
    check=False
    with open(skip_out, 'w') as HILLS_skipped:
        for line in open(files, 'r').readlines():
            if not line[0] in ['#', '@']:
                    if np.round(np.round(float(line.split()[0]), 3),3).is_integer():
                        count+=1            
                        check=False
                        HILLS_skipped.write(line)
                    if (count/1000).is_integer() and check == False:
                        check=True

def write_file(CV,d1x,d1y,energy,site):
        with open('1D_landscape_site-'+str(site), 'w') as landscape:
            for line in range(len(CV)):
                landscape.write(str(CV[line])+'\t'+str(energy[line])+'\t'+str(d1x[line])+'\t'+str(d1y[line])+'\n')

def readfes(files):
    x, y, z, e=[],[],[],[]
    start_fes=time.time()
    fes = open(files, 'r')
    for line in fes.readlines():
        if not line[0] in ['#', '@'] and len(line)>1:
            x.append(float(line.split()[0]))
            y.append(float(line.split()[1]))
            z.append(float(line.split()[2]))
            e.append(float(line.split()[3]))
    print('reading in '+files.split('.')[0]+' took:', np.round(time.time() - start_fes, 2))
    fes.close()
    return np.array(x),np.array(y),np.array(z),np.array(e)

def clockwiseangle_and_distance(point):
    origin = [0,0]
    refvec = [0, 1]
    # Vector between point and the origin: v = p - o
    vector = [point[0]-origin[0], point[1]-origin[1]]
    # Length of vector: ||v||
    lenvector = math.hypot(vector[0], vector[1])
    # If length is zero there is no angle
    if lenvector == 0:
        return -math.pi, 0
    # Normalize vector: v/||v||
    normalized = [vector[0]/lenvector, vector[1]/lenvector]
    dotprod  = normalized[0]*refvec[0] + normalized[1]*refvec[1]     # x1*x2 + y1*y2
    diffprod = refvec[1]*normalized[0] - refvec[0]*normalized[1]     # x1*y2 - y1*x2
    angle = math.atan2(diffprod, dotprod)
    # Negative angles represent counter-clockwise angles so we need to subtract them 
    # from 2*pi (360 degrees)
    if angle < 0:
        return 2*math.pi+angle, lenvector
    # I return first the angle because that's the primary sorting criterium
    # but if two vectors have the same angle then the shorter distance should come first.
    return angle, lenvector

def bulk_val(x,y,z): 
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
            if np.sqrt(((sorted_coord[i-1][0]-0)**2)+((sorted_coord[i-1][1]-0)**2)) > 2 and run==False:
                run=True
                sorted_x, sorted_y = [sorted_coord[i-1][0]], [sorted_coord[i-1][1]]
            if run==True:
                if np.sqrt(((sorted_coord[i][0]-sorted_x[-1])**2)+((sorted_coord[i][1]-sorted_y[-1])**2)) < 0.5 and run==True:
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
            maxd=1
            while l <= maxd:   ### max distance from outer line 
                s-=0.01
                coords=[np.round(coord[0]*s, 3),np.round(coord[1]*s, 3)]
                l = np.sqrt(((coord[0]-coords[0])**2)+((coord[1]-coords[1])**2))
                if l <= maxd-0.5:   ### min distance form outer line 
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
    #   plt.scatter(sorted_X_l, sorted_Y_l)
    #   plt.scatter(sorted_X_s, sorted_Y_s)
    #   plt.scatter(x[center][bulk_values], y[center][bulk_values])
    # plt.show()

    if len(bulk) == 0:
        bulk=0

    # print('done bulk ')
    bulk = np.round(float(np.nanmean(bulk)),2)
    # print(bulk)
    return bulk, sorted_X_s, sorted_Y_s, sorted_X_l, sorted_Y_l

def hills_converge(param):

    labels=25   #energy label
    title_height=0.75 #0.75
    labels_size=35

    lab=[]
    time, dx, dy,gau = readhills(param['HILLS_skipped'])
    time_bulk ,bulk = read_bulk(param['bulk_values'],time, dx, dy,gau)

    if 'ring_location' not in param:
        plot_numbers=2
    else:
        plot_numbers=2+len(param['ring_location'])
    lim_ux=max(time)+2
    if os.path.exists('energies_time'):
        sites = np.genfromtxt('energies_time')
        time_energy=np.arange(0,len(sites[:,0]), 1)*0.25
    start = np.argmax(ave(bulk, 10)<np.max(gau)*0.05)
    plt.figure(1, figsize=(20,30))

    plt.subplot(plot_numbers,1,1)
    plt.title('Raw gaussian height' ,  fontproperties=font1, fontsize=40,y=1.05)
    # plt.axvline(ave(time_bulk, 10)[start],ymin=0, ymax=0.70, linewidth=8,color='k')
    plt.plot(time,gau, color='blue',linewidth=4)
    plt.plot(ave(time, 10), ave(gau, 10), color='red',linewidth=4)
    plt.yticks(np.arange(0, 1.3,0.4), fontproperties=font1,  fontsize=30)#
    plt.xticks(np.arange(0, 200,10), fontproperties=font1,  fontsize=30)#
    plt.ylim(-0.1,1.2);plt.xlim(-2, lim_ux)
    plt.tick_params(axis='both', which='major', width=3, length=5, labelsize=30, direction='in', pad=10, right=False, top=False,labelbottom=False)
    plt.ylabel('Hills height \n(kJ mol$^{-1}$)', fontproperties=font2,fontsize=labels_size)

    plt.subplot(plot_numbers,1,2)
    plt.title('Bulk gaussian height' ,  fontproperties=font1, fontsize=40,y=title_height)
    plt.axvline(ave(time_bulk, 10)[start],ymin=0, ymax=0.70, linewidth=8,color='k')
    plt.plot(time_bulk ,bulk, color='blue',linewidth=4)
    plt.plot(ave(time_bulk, 10), ave(bulk, 10), color='red',linewidth=4)
    plt.yticks(np.arange(0, 1.3,0.4), fontproperties=font1,  fontsize=30)#
    plt.xticks(np.arange(0, 200,10), fontproperties=font1,  fontsize=30)#
    plt.ylim(-0.1,1.2);plt.xlim(-2, lim_ux)
    plt.tick_params(axis='both', which='major', width=3, length=5, labelsize=30, direction='in', pad=10, right=False, top=False,labelbottom=False)
    plt.ylabel('Hills height \n(kJ mol$^{-1}$)', fontproperties=font2,fontsize=labels_size) 
    # plt.xlabel('Time ($\mu$s)', fontproperties=font2,fontsize=30)
    if param['circle_plot'] or param['ellipse_plot']:
        for val, fig_num in enumerate(range(3,2+len(param['ring_location'])*2,2)): 
            if param['circle_plot']:
                center=np.where(np.sqrt(((param['ring_location'][val][0]-dx)**2)+((param['ring_location'][val][1]-dy)**2)) <= param['circle_area'][val])
            elif param['ellipse_plot']:
                center=ellipse_check_point(dx, dy, param['ring_location'][val], param['ellipse_width'][val], param['ellipse_height'][val], param['ellipse_angle'][val])





            # for time_interval in range(0, 130, 10):
            #     try:
            #         average_ind=np.where(np.logical_and(time_energy>ave(time_bulk, 10)[start],time_energy<ave(time_bulk, 10)[start]+time_interval))
            #     except:
            #         average_ind=np.where(time_energy>ave(time_bulk, 10)[start])
            #         break
            plt.subplot(2+len(param['ring_location']),1,val+3)
            plt.axvline(ave(time_bulk, 10)[start],ymin=0, ymax=0.70, linewidth=8,color='k')

            plt.title('Site '+str(val) ,  fontproperties=font1, fontsize=40,y=title_height)#+' gaussian height'

            plt.scatter(time[center],gau[center], s=100, color='blue')
            plt.scatter(ave(time[center], 10), ave(gau[center], 10), s=25, alpha=0.3, color='red')
            plt.yticks(np.arange(0, 1.21,0.4), fontproperties=font1,  fontsize=35)
            plt.xticks(np.arange(0, 200,10), fontproperties=font1,  fontsize=35)#
            plt.ylim(-0.1,1.2);plt.xlim(-2, lim_ux)
            # plt.tick_params(axis='both', which='major', width=3, length=5, labelsize=30, direction='in', pad=10, right=False, top=False,labelbottom=False)
            plt.ylabel('Hills height \n(kJ mol$^{-1}$)', fontproperties=font2,fontsize=labels_size) 

            # plt.subplot(2+len(ring_location)*2,1,fig_num+1)
            # # plt.title('site '+str(s+1)+' free energy' ,  fontproperties=font1, fontsize=35,y=0.6)
            # plt.axvline(ave(time_bulk, 10)[start],ymin=0, ymax=0.70, linewidth=8,color='k')
            # plt.scatter(time_energy, sites[:,rin], s=100, color='blue')
            # plt.plot(time_energy, sites[:,rin], color='red')#,label=str(np.round(np.mean(sites[:,rin][average_ind]),1))+'  '+str(np.round(np.std(sites[:,rin][average_ind]),1)))
            # plt.scatter(time_energy, sites[:,rin], s=100)
            # plt.yticks(np.arange(-100, 21,20), fontproperties=font1,  fontsize=30)
            # plt.xticks(np.arange(0, 200,10), fontproperties=font1,  fontsize=30)#
            # plt.ylim(lim_ly,20);plt.xlim(-2, lim_ux)
            # if val+3 != 2+len(ring_location):
            #   plt.tick_params(axis='both', which='major', width=3, length=5, labelsize=30, direction='in', pad=10, right=False, top=False,labelbottom=False)
            # else:
            #   plt.tick_params(axis='both', which='major', width=3, length=5, labelsize=30, direction='in', pad=10, right=False, top=False)

            # plt.ylabel('Energy \n(kJ mol$^{-1}$)', fontproperties=font2,fontsize=labels_size) 
            # plt.annotate(str(int(np.round(np.mean(sites[:,rin][average_ind]),0)))+' $\pm$ '+str(int(np.round(np.std(sites[:,rin][average_ind]),0)))+' kJ mol$^{-1}$', xy=(label_loc_x, label_loc_y), size =labels, bbox=bbox, ha="right", va="top")
            # rin+=1
    plt.xlabel('Time ($\mu$s)', fontproperties=font2,fontsize=labels_size)
    plt.subplots_adjust( top=0.955, bottom=0.075, left=0.12,right=0.97, wspace=0.4, hspace=0.18)

    plt.savefig('sites.png', dpi=300)


    plt.show()

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


def final_frame(param, error, save_plot):
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
        x,y,z,e=readfes(param['fes'])    ### reads in energy landscape
    #### fetch bulk area    
        if os.path.exists(param['bulk_values']):  ### reads bulk value file to get bulk area
            coord=np.genfromtxt(param['bulk_values'], autostrip=True)
            sorted_X_s, sorted_Y_s, sorted_X_l, sorted_Y_l=coord[:,0],coord[:,1],coord[:,2],coord[:,3]
            shuffledxy = np.stack((x,y), axis=-1)
            bulk, sorted_X_s, sorted_Y_s, sorted_X_l, sorted_Y_l = find_bulk(x,y,z,shuffledxy,sorted_X_s, sorted_Y_s, sorted_X_l, sorted_Y_l)
        else:   ### finds bulk area from scratch (much slower)
            print('finding bulk area')
            bulk, sorted_X_s, sorted_Y_s, sorted_X_l, sorted_Y_l = bulk_val(x,y,z)

        z[ z==0 ] = np.nan    ### removes unsampled area by converting to nan
        z=np.clip(z-bulk,np.nanmin(z)-bulk,param['energy_max'])    ### references energy lanscape to bulk and caps the FES max to 20 kJ/mol

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

        ### allows manually defined maxz and minz
        maxz =np.nanmax(Z)  
        maxz=10
        minz=find_min(np.nanmin(Z))
        # minz=np.round(np.nanmin(Z),-1)-step
        ### np.arange(np.round(minz,0)-(step/2), maxz+1, step)
        # cax = ax1.contourf(X*invert,Y,Z,np.arange(np.round(minz,0)-(step/2), maxz+1, step), cmap=plt.get_cmap('coolwarm_r'),norm=MidpointNormalize(midpoint=0.), alpha=0.25, vmin = minz, vmax = maxz )# , alpha=0.25)
        # con = ax1.contour(X*invert,Y,Z,np.arange(np.round(minz,0)-(step/2), maxz+1, step), linewidths=8, cmap=plt.get_cmap('coolwarm_r'), linestyles='-',norm=MidpointNormalize(midpoint=0.), alpha=1, vmin = minz, vmax = maxz )#, alpha=1)#colors='k'# cax = ax1.contourf(X,Y,Z,np.arange(-100, 15, 5), cmap=cm.coolwarm, vmin = minz, vmax = 15 , alpha=1)

    ### FES shading and contours
        cax = ax1.contourf(X*param['invert'],Y,Z,np.arange(np.round(minz,0), 0.1, param['step']), cmap=plt.get_cmap('coolwarm_r'),norm=MidpointNormalize(midpoint=0.), alpha=0.25, vmin = minz, vmax = maxz )# , alpha=0.25)
        con = ax1.contour(X*param['invert'],Y,Z,np.append(np.arange(minz, 0, param['step']*2),10), linewidths=8, cmap=plt.get_cmap('coolwarm_r'), linestyles='-',norm=MidpointNormalize(midpoint=0.), alpha=1, vmin = minz, vmax = maxz )#, alpha=1)#colors='k'# cax = ax1.contourf(X,Y,Z,np.arange(-100, 15, 5), cmap=cm.coolwarm, vmin = minz, vmax = 15 , alpha=1)

    ### colourbar   np.arange(np.round(minz,-1), maxz+1, step*2))
        cbar = fig1.colorbar(cax, ticks=np.append(np.arange(np.round(minz,0), 0.1, param['colour_bar_tick']*4),param['colour_bar_tick']))#.set_alpha(1)
        ticklabs = cbar.ax.get_yticklabels()
        cbar.ax.set_yticklabels(ticklabs, ha='right', va='center', fontsize=60)
        cbar.set_label('Free energy (kJ mol$^{-1}$)',fontsize=60, labelpad=40)
        cbar.ax.yaxis.set_tick_params(pad=200)
        cbar.set_alpha(1)
        cbar.draw_all()
        text_params = {'ha': 'center', 'va': 'center', 'family': 'sans-serif','fontweight': 'bold'}

    ### FES highlights circle or ellipse
        if param['circle_plot']:
            # circle = plt.Circle((0,0, area[val],linewidth=10,edgecolor='k',facecolor='none', zorder=2)
            for val, ring in enumerate(param['ring_location']):
                circle = plt.Circle((ring[0],ring[1]), param['circle_area'][val],linewidth=10,edgecolor='k',facecolor='none', zorder=2)
                ax1.add_artist(circle)
        if param['ellipse_plot']:
            for val, ring in enumerate(param['ring_location']):
            # ellipse = Ellipse(xy=(-1.69,75), width=0.4, height=60, angle=0.25,linewidth=10,edgecolor='k',facecolor='none', zorder=2) # original
                ellipse = patches.Ellipse(xy=(ring[0],ring[1]), width=param['ellipse_width'], height=param['ellipse_height'], angle=param['ellipse_angle'],linewidth=10,edgecolor='k',facecolor='none', zorder=2) # original
                ax1.add_artist(ellipse)

    ### Tick parameters
        
        ax1.set_xlabel(param['CV1'], fontproperties=font1,  fontsize=80);ax1.set_ylabel(param['CV2'], fontproperties=font1,  fontsize=80)
        plt.xticks(np.arange(-180, 180.1,param['interval_x']), fontproperties=font1,  fontsize=80)
        plt.yticks(np.arange(-180, 180.1,param['interval_y']), fontproperties=font1,  fontsize=80)#
        ax1.tick_params(axis='both', which='major', width=3, length=5, labelsize=80, direction='out', pad=10)

    ### plot bulk area in dotted lines
        if param['bulk_outline']:
            plt.plot(sorted_X_l, sorted_Y_l, color='k', linestyle='--', linewidth=6)
            plt.plot(sorted_X_s, sorted_Y_s, color='k', linestyle='--', linewidth=6)

    ### 1D stuff bars
        if param['1d']:
            d_landscape, x_landscape, y_landscape, z_landscape =[],[],[],[]
            for val, x_points_start in enumerate(param['x_points']):
                distance,d1x,d1y,d1z,point_lower_x,point_lower_y,point_upper_x,point_upper_y=strip_1d(x,y,z, x_points_start, param['y_points'][val], param['search_width'], param['invert'])
                plt.plot([point_lower_x[0],point_lower_x[-1]],[point_lower_y[0],point_lower_y[-1]], color='red',linewidth=10, zorder=2)
                plt.plot([point_upper_x[0],point_upper_x[-1]],[point_upper_y[0],point_upper_y[-1]], color='red',linewidth=10, zorder=2)
                plt.plot(d1x,d1y, color='k',linewidth=10, zorder=2)
                d_landscape.append(distance)
                x_landscape.append(d1x)
                y_landscape.append(d1y)
                z_landscape.append(d1z)
                if save_plot != None:
                    write_file(distance,d1x,d1y,d1z, val+1)
    ### plot limits
        
        minx, maxx=min(x)-0.1 ,max(x)+0.1
        # if args.p in ['lsp','lgt']:
        #   minx, maxx=0 ,max(x)+0.1   ### 

        miny, maxy=min(y)-0.1,max(y)+0.1
        plt.xlim(minx,maxx);plt.ylim(miny,maxy)
        plt.subplots_adjust(top=0.972, bottom=0.13,left=0.168,right=0.90, hspace=0.2, wspace=0.2)

        if save_plot != None:
            plt.savefig(save_plot+'.png', dpi=300)
        plt.show()

        if param['1d']:
            fig2 = plt.figure(2, figsize=(35,20))### error plot   [0,2.5,5,7.5,10,12.5,15]
            for val, landscape in enumerate(d_landscape):
                plt.plot(landscape, z_landscape[val], linewidth=5, label='site '+str(val+1))
            plt.tick_params(axis='both', which='major', width=3, length=5, labelsize=15, direction='in', pad=10, right=False, top=False)
            plt.xlabel('Distance (nm)', fontproperties=font2,fontsize=15);plt.ylabel('Energy (kJ mol$^{-1}$)', fontproperties=font2,fontsize=15) 
            plt.legend(prop={'size': 10}, bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=len(d_landscape), mode="expand", borderaxespad=0.)
            if save_plot != None:
                plt.savefig(save_plot+'_1D.png', dpi=300)
            plt.show()
    else:
#### Error analysis plot

        fig1 = plt.figure(1, figsize=(35,20))### 
        step = 2.5
        ax = fig1.add_subplot(111, aspect='equal')
        im1 = plt.imread(picture)
        implot1 = plt.imshow(im1,aspect='equal', extent=(picture_loc[0],picture_loc[1],picture_loc[2],picture_loc[3]), alpha=1)
        cax = ax.contourf(X*invert,Y,E, np.arange(0,np.nanmax(e)+step,step),cmap=plt.get_cmap('coolwarm'),norm=MidpointNormalize(midpoint=0.), alpha=0.25, vmin = np.nanmin(e), vmax = np.nanmax(e))# , alpha=0.25)
        cone = ax.contour(X*invert,Y,E, np.arange(0,np.nanmax(e)+step,step),linewidths=8, cmap=plt.get_cmap('coolwarm'), linestyles='-',norm=MidpointNormalize(midpoint=0.), alpha=1, vmin = np.nanmin(e), vmax = np.nanmax(e) )#, alpha=1)#colors='k'# cax = ax1.contourf(X,Y,Z,np.arange(-100, 15, 5), cmap=cm.coolwarm, vmin = minz, vmax = 15 , alpha=1)
        conz = ax.contour(X*invert,Y,Z,[5], linewidths=8, cmap=plt.get_cmap('coolwarm_r'), linestyles='-',norm=MidpointNormalize(midpoint=0.), alpha=1, vmin = minz, vmax = maxz )#, alpha=1)#colors='k'# cax = ax1.contourf(X,Y,Z,np.arange(-100, 15, 5), cmap=cm.coolwarm, vmin = minz, vmax = 15 , alpha=1)

    #### colourbar   np.arange(np.round(minz,-1), maxz+1, step*2))
        cbar = fig1.colorbar(cax, ticks=np.arange(0,np.nanmax(e)+step,step))#np.append(np.arange(np.round(minz,0), -2.5, step*2),5))#.set_alpha(1)
        ticklabs = cbar.ax.get_yticklabels()
        cbar.ax.set_yticklabels(ticklabs, ha='right', va='center', fontsize=60)
        cbar.set_label('Free energy (kJ mol$^{-1}$)',fontsize=60, labelpad=40)
        cbar.ax.yaxis.set_tick_params(pad=200)
        cbar.set_alpha(1)
        cbar.draw_all()
        # print(np.nanmin(E), np.nanmax(E))
        for val, ring in enumerate(ring_location):
            circle = plt.Circle((ring[0],ring[1]), area[val],linewidth=10,edgecolor='k',facecolor='none', zorder=2)
            ax.add_artist(circle)
        plt.xticks(np.arange(-10, 10.1,interval), fontproperties=font1,  fontsize=80)
        plt.yticks(np.arange(-10, 10.1,interval), fontproperties=font1,  fontsize=80)#
        plt.xlim(min(x)-0.1,max(x)+0.1);plt.ylim(min(y)-0.1,max(y)+0.1)
        plt.subplots_adjust(top=0.972, bottom=0.13,left=0.168,right=0.90, hspace=0.2, wspace=0.2)
        if save_plot != None:
            plt.savefig(save_plot+'_error.png', dpi=300)
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
        x,y,z, e=readfes(param['prefix']+str(fes)+'.dat')
        start_time = time.time()
        bulk, sorted_X_s, sorted_Y_s, sorted_X_l, sorted_Y_l = bulk_val(x,y,z)
        output=[fes,bulk]
        if param['circle_plot'] or param['ellipse_plot']:
            if len(param['ring_location']) > 0:
                for val, ring in enumerate(param['ring_location']):
                    if param['circle_plot']:
                        minima=np.where(np.sqrt(((ring[0]-x)**2)+((ring[1]-y)**2)) <= param['circle_area'][val])
                    elif param['ellipse_plot']:
                        minima=ellipse_check_point(x, y, ring, param['ellipse_width'][val], param['ellipse_height'][val], param['ellipse_angle'][val])
                    output.append(np.round(min(z[minima])-bulk,4))
                return output
            else:
                sys.exit('No locations specified')
        else:
            sys.exit('Ellipse or circle not been selected')

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
    start_time = time.time()
    x,y,z, e=readfes(param['prefix']+str(fes)+'.dat')
    shuffledxy = np.stack((x,y), axis=-1)
    bulk, sorted_X_s, sorted_Y_s, sorted_X_l, sorted_Y_l = find_bulk(x,y,z,shuffledxy,sorted_X_s, sorted_Y_s, sorted_X_l, sorted_Y_l)
    z[ z==0 ] = np.nan
    z=np.clip(z-bulk,np.nanmin(z)-bulk,20)
    return x,y,z

def get_frames(run):
    param =parameters()
    os.system('mkdir analysis')
    os.chdir('analysis')
    xtc_files=[]
    for filename in os.listdir('../'):
        if filename.endswith(".xtc"):
            xtc = re.search('\.(.*)\.', filename)[1]
            xtc_files.append(xtc)
    pool = mp.Pool(mp.cpu_count())

    test = pool.map_async(gromacs, ['gmx distance -f ../*'+xtc+'.xtc -s ../*tpr -n ../../build/index.ndx -oxyz '+xtc+'.xvg -select \'cog of group \"trans\" plus cog of group \"pip2_head\"\'' for xtc in xtc_files]).get()
    pool.join
    
    for val, xtc in enumerate(xtc_files):
        file_out=np.genfromtxt(xtc+'.xvg', autostrip=True, comments='@',skip_header=13)
        for ring_num, ring in enumerate(ring_location):
            loc=np.where(np.sqrt(((float(ring[0])-file_out[:,1])**2)+((float(ring[1])-file_out[:,2])**2)) <= area[ring_num])
            time_loc=file_out[:,0][loc]
            start_time = time.time()
            test = pool.map_async(gromacs, ['echo 0 | gmx trjconv -pbc res -f ../*'+xtc+'.xtc -s ../*tpr -b '+str(time_stamp)+' -e '+str(time_stamp)+' -o ring_'+str(ring_num+1)+'_'+xtc+'_'+str(int(time_stamp))+'_ind.xtc' for time_stamp in time_loc]).get()
            pool.join

            print(time.time()-start_time)
            out=gromacs('gmx trjcat -f ring_'+str(ring_num+1)+'_'+xtc_files[val]+'_*.xtc -o all_ring_'+str(ring_num+1)+'_'+xtc_files[val]+'.xtc' )

    os.system('rm r*_part*')
    for ring, ring_loc in enumerate(ring_location):
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


def strip_1d(x,y,z, x_points, y_points, distance, invert):
    length_CV=np.sqrt((x_points[0]-x_points[1])**2+(y_points[0]-y_points[1])**2)
    number_points=length_CV/0.005
    CV=np.linspace(0,int(length_CV),int(number_points),endpoint=False)
    point_lower_x, point_lower_y,point_upper_x,point_upper_y=np.array([]),np.array([]),np.array([]),np.array([])

    if x_points[0]!=x_points[1] and y_points[0]!=y_points[1]:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_points,y_points)
        perp_slope=(-1/slope)
        perp_intercept=(perp_slope*x_points[0]-y_points[0])*-1
        if x_points[1] < x_points[0]:
            line_x = np.array(x_points[0]-CV*np.sqrt(1/(1+slope**2)))
            line_y = np.array(y_points[0]-CV*slope*np.sqrt(1/(1+slope**2)))
        else:
            line_x = np.array(x_points[0]+CV*np.sqrt(1/(1+slope**2)))
            line_y = np.array(y_points[0]+CV*slope*np.sqrt(1/(1+slope**2)))
        for val, x_val in enumerate(line_x):
            y_val=line_y[val]
            perp_intercept=(perp_slope*x-y)*-1
            point_lower_x= np.append(point_lower_x, x_val+dx(distance,perp_slope))
            point_lower_y= np.append(point_lower_y, y_val+dy(distance,perp_slope))
            point_upper_x= np.append(point_upper_x, x_val-dx(distance,perp_slope))
            point_upper_y= np.append(point_upper_y, y_val-dy(distance,perp_slope))
    # return point_lower_x*invert,point_lower_y,point_upper_x*invert,point_upper_y
    else:
        if x_points[0]==x_points[1]:
            line_x = np.array(x_points[0]+np.zeros(len(CV)))
            line_y = np.array(y_points[0]+CV)
            for val, y_val in enumerate(CV):
                point_lower_x= np.append(point_lower_x, x_points[0]-distance)
                point_lower_y= np.append(point_lower_y, y_points[0]+y_val)
                point_upper_x= np.append(point_upper_x, x_points[0]+distance)
                point_upper_y= np.append(point_upper_y, y_points[0]+y_val)
        if y_points[0]==y_points[1]:
            line_x = np.array(x_points[0]+CV)
            line_y = np.array(y_points[0]+np.zeros(len(CV)))
            for val, x_val in enumerate(CV):
                point_lower_x= np.append(point_lower_x, x_points[0]+x_val)
                point_lower_y= np.append(point_lower_y, y_points[0]-distance)
                point_upper_x= np.append(point_upper_x, x_points[0]+x_val)
                point_upper_y= np.append(point_upper_y, y_points[0]+distance)               
    shuffledxy = np.stack((x,y), axis=-1)
    pool = mp.Pool(mp.cpu_count())
    xyz = pool.starmap_async(points_1d, [(x,y,z,point_lower_x,point_lower_y,point_upper_x,point_upper_y,shuffledxy,frame, line_x, line_y) for frame in range(len(line_x)-1)]).get()
    pool.close
    xyz = np.array([col for col in xyz if col != None])
    return np.sqrt(((xyz[:,3]-float(line_x[0]))**2)+((xyz[:,4]-float(line_y[0]))**2)), xyz[:,0]*invert, xyz[:,1], xyz[:,2],point_lower_x,point_lower_y,point_upper_x,point_upper_y


def average_1d(param, fes_frame):   ### not tested
    start_time = time.time()
    x,y,z,e=readfes(param['prefix']+str(fes_frame)+'.dat')
    try:  ### reads bulk value file to get bulk area
        coord=np.genfromtxt('bulk_values', autostrip=True)
        sorted_X_s, sorted_Y_s, sorted_X_l, sorted_Y_l=coord[:,0],coord[:,1],coord[:,2],coord[:,3]
        shuffledxy = np.stack((x,y), axis=-1)
        bulk, sorted_X_s, sorted_Y_s, sorted_X_l, sorted_Y_l = find_bulk(x,y,z,shuffledxy,sorted_X_s, sorted_Y_s, sorted_X_l, sorted_Y_l)
    except:   ### finds bulk area from scratch (much slower)
        print('finding bulk area')
        bulk, sorted_X_s, sorted_Y_s, sorted_X_l, sorted_Y_l = bulk_val(x,y,z)
        # z[ z==0 ] = np.nan
        z=np.clip(z-bulk,np.nanmin(z)-bulk,20)
    for val, x_points_start in enumerate(param['x_points']):
        if not os.path.exists(param['1d_location']+'/1D_landscape_site-'+str(val+1)+'_'+str(fes_frame)):
            distance,d1x,d1y,d1z,point_lower_x,point_lower_y,point_upper_x,point_upper_y = strip_1d(x,y,z, x_points_start, param['y_points'][val], param['search_width'], param['invert'])
            with open(param['1d_location']+'/1D_landscape_site-'+str(val+1)+'_'+str(fes_frame), 'w') as landscape:
                for line in range(len(distance)):
                    landscape.write(str(np.round(distance[line],3))+'\t'+str(d1z[line])+'\t'+str(np.round(d1x[line],5))+'\t'+str(np.round(d1y[line],5))+'\n')
            landscape.close()
    # print(fes_frame, np.round(time.time()-start_time, 2))

def bootstrap_1d(param):
    for site in range(1,len(param['x_points'])+1):
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

                bulk=np.where(np.logical_and(data[:,0]>param['cutoff'][site-1]-param['cut'], data[:,0]<param['cutoff'][site-1]))
                bulk_val=(np.mean(data[:,1][bulk]))
                data[:,1]=data[:,1]-bulk_val
                if init:
                    init=False
                    landscapes = {str(np.round(CV,3)): [] for CV in np.linspace(0,np.max(data[:,0])+0.05,(np.max(data[:,0])+0.05)*200,endpoint=False)}
                for val,CV in enumerate(data[:,0]):
                    landscapes[str(CV)].append(data[:,1][val])
            else:
                pass
        for CV in np.linspace(0,3.5,700,endpoint=False):
            if len(landscapes[str(np.round(CV,3))]) >2 and CV < param['cutoff'][site-1] and CV > param['min_cv'][site-1]:
                bs_sample = np.random.choice(landscapes[str(np.round(CV,3))], size=10000)
                energy.append([np.round(CV,3), np.mean(bs_sample), np.std(bs_sample)])
        energy=np.array(energy)
        plt.figure(1, figsize=(20,20))
        plt.subplot(len(param['ring_location']),1,site)      
        cutoff_new = float(param['cutoff'][site-1]-energy[:,0][np.where(energy[:,1]==np.min(energy[:,1]))])
        energy[:,0] = energy[:,0] - energy[:,0][np.where(energy[:,1]==np.min(energy[:,1]))]
        cutoff_max = energy[:,0][np.where(energy[:,1]==np.min(energy[:,1]))]

        if cutoff_max < np.max(energy[:,0]):
            cutoff_max=np.max(energy[:,0])
        plt.title('Site '+str(site) ,  fontproperties=font1, fontsize=40,y=param['title_height'])#+' gaussian height'

        plt.plot(energy[:,0], energy[:,1], linewidth=3, color='red')
        plt.fill_between(energy[:,0], energy[:,1]-energy[:,2], energy[:,1]+energy[:,2], alpha=0.3, facecolor='black')
        error=np.mean(energy[:,2][np.where(np.logical_and(energy[:,0]>cutoff_new-param['cut'],energy[:,0]<cutoff_new))])+energy[:,2][np.where(energy[:,1]==np.min(energy[:,1]))]
        plt.annotate(str(int(np.round(np.min(energy[:,1]),0)))+' $\pm$ '+str(int(np.round(error,0)))+' kJ mol$^{-1}$', xy=(cutoff_max-0.1, param['lab_y']), size =25, bbox=bbox, ha="right", va="top")
        plt.yticks(np.arange(-500, param['energy_max']+1,param['step_yaxis']), fontproperties=font1,  fontsize=35)
        plt.xticks(np.arange(-3, 10,0.5), fontproperties=font1,  fontsize=35)#
        if site<len(param['ring_location']):
            plt.tick_params(axis='both', which='major', width=3, length=5, labelsize=35, direction='in', pad=10, right=False, top=False,labelbottom=False)
        else:
            plt.tick_params(axis='both', which='major', width=3, length=5, labelsize=35, direction='in', pad=10, right=False, top=False)
            plt.xlabel('CV (nm)', fontproperties=font2,fontsize=param['labels_size']) 

        plt.ylabel('Energy \n(kJ mol$^{-1}$)', fontproperties=font2,fontsize=param['labels_size']) 
        # plt.xlim(0,cutoff)
        plt.xlim(np.min(energy[:,0]),cutoff_max)
        plt.ylim(np.min(energy[:,1])*1.1,20)
    plt.savefig(param['1d_location']+'/energy_landscape_sites_1d.png', dpi=300)
    plt.show()


def plot_CV(param):
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
    for hill_val, hills_file in enumerate(xy):
        ax1 = fig1.add_subplot(5,4, hill_val+1, aspect='equal')
        plt.title(str(hill_val+1) ,  fontproperties=font1, fontsize=20,y=0.9)
        if param['circle_plot'] or param['ellipse_plot']:
            for val, ring in enumerate(param['ring_location']):
                if param['circle_plot']:
                    circle = plt.Circle((ring[0],ring[1]), param['circle_area'][val],linewidth=2,edgecolor='k',facecolor='none', zorder=2)
                    ax1.add_artist(circle)
                if param['ellipse_plot']:
                    ellipse = patches.Ellipse(xy=(ring[0],ring[1]), width=param['ellipse_width'], height=param['ellipse_height'], angle=param['ellipse_angle'],linewidth=10,edgecolor='k',facecolor='none', zorder=2) 
                    ax1.add_artist(ellipse)

        ran=np.linspace(0,1,len(xy[hill_val][hill_val:,0]))
        ax1.scatter(xy[hill_val][hill_val:,0], xy[hill_val][hill_val:,1],s=1,c = cm.coolwarm(ran))
        plt.yticks(np.arange(-10, 10,2), fontproperties=font1,  fontsize=20)#
        plt.xticks(np.arange(-10, 10,2), fontproperties=font1,  fontsize=20)#
        rows=[1,5,9,13,17]
        lef, bot=False,False    
        if hill_val+1 in rows:
            plt.ylabel(param['CV1'], fontproperties=font2,fontsize=20) 
            lef=True
        if hill_val+1 in np.arange(len(hills_files), len(hills_files)-4,-1):
            plt.xlabel(param['CV1'], fontproperties=font2,fontsize=20)
            bot=True  
        ax1.tick_params(axis='both', which='major', width=3, length=5, labelsize=20, direction='in', pad=10 , labelbottom=bot, labelleft=lef,right=True, top=True, left=True, bottom=True)
        plt.subplots_adjust(left=0.1, wspace=0, hspace=0, top=0.975, bottom=0.08)
    plt.savefig('hills_trace.png', dpi=300)
    plt.show()

def multi_read_hills(hill_val, hills_file):
    hills=[]
    for line in open(hills_file, 'r').readlines():
        if not line[0] in ['#', '@']:
            if np.round(np.round(float(line.split()[0]), 3),3).is_integer():
                hills.append([float(line.split()[1]), float(line.split()[2])])
    hills = np.array(hills)
    if len(hills[:,0]) != len(hills[:,1]):
        sys.exit('error with HILLS file: '+hill_val)
    return hills


def timecourse():
    plt.close()
    min_energy=0
    count=0
    run_ylim=False
    plt.figure(1, figsize=(20,10))
    for site in range(1,len(param['x_points'])+1):
        count+=1
        if site==len(param['x_points']):
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
                bulk=np.where(np.logical_and(data[:,0]>param['cutoff'][site-1]-cut, data[:,0]<param['cutoff'][site-1]))
            except:
                pass
                print(site, f)
                print(len(cutoff))
                print(cutoff[site-1]-cut, cutoff[site-1])

            bulk_val=(np.mean(data[:,1][bulk]))
            data_ori=data.copy()
            data[:,1]=data[:,1]-bulk_val
            energy_min=np.append(energy_min, np.nanmin(data[:,1]))
            if site==3 and  str(np.nanmin(data[:,1])) == 'nan':
                print(data_ori[:,1] ,data[:,1], bulk_val,  bulk)


        plt.plot(np.array(number_range)*0.25, energy_min, label='site: '+str(site), linewidth=4)

        plt.yticks(np.arange(-100, 21,step/2), fontproperties=font1,  fontsize=30)
        plt.xticks(np.arange(0, 300,20), fontproperties=font1,  fontsize=30)#
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

    plt.savefig('energy_timecourse_'+args.p+'.png', dpi=300)
    # plt.show()

# parser = argparse.ArgumentParser()
# parser.add_argument('-f', help='function',metavar='plot',type=str, choices= ['sort', 'skip', 'plot', 'boot', 'concat','strip', 'converge', 'frames', '1d_time', 'cv_plot', 'timecourse'], required=True)
# parser.add_argument('-input', help='name of input variables',metavar='protein',type=str, required=True)
# parser.add_argument('-save', help='save name',metavar='FES',type=str)
# parser.add_argument('-d1', help='plots 1d landscape', action='store_true')
# parser.add_argument('-bulk', help='plots bulk outline', action='store_true')
# parser.add_argument('-error', help='plots error instead of energy', action='store_true')
# parser.add_argument('-test', help='runs stripped down version of script', action='store_true')
# parser.add_argument('-s', help='start',metavar='1',type=int)
# parser.add_argument('-e', help='end',metavar='10',type=int)
# #parser.add_argument('-tpr', help='do not make tpr files', action='store_false')
# args = parser.parse_args()
# options = vars(args)

# param = parameters()
# ### working
# if args.f == 'sort':
#     os.system("awk \'{print $NF,$0}\' HILLS | sort -n | cut -f2- -d\' \' > HILLS_sorted")

# if args.f == 'skip':
#     if not os.path.exists('HILLS_sorted'):
#         os.system("awk \'{print $NF,$0}\' HILLS | sort -n | cut -f2- -d\' \' > HILLS_sorted")
#     skip_hills('HILLS_sorted')

# if args.f == 'plot':
#     final_frame()

# #### average z across frames
# if args.f== 'concat':
#     start, end=args.s,args.e
#     if os.path.exists('bulk_values'):
#         coord=np.genfromtxt('bulk_values', autostrip=True)
#         sorted_X_s, sorted_Y_s, sorted_X_l, sorted_Y_l=coord[:,0],coord[:,1],coord[:,2],coord[:,3]
#     else:
#         x,y,z,e=readfes(param['prefix']+str(end)+'.dat')
#         bulk, sorted_X_s, sorted_Y_s, sorted_X_l, sorted_Y_l = bulk_val(x,y,z)
#     if os.path.exists('landscapes_done_temp'):
#         os.system('rm -r landscapes_done_temp')
#     os.mkdir('landscapes_done_temp')    
#     pool = mp.Pool(mp.cpu_count())
#     xyz = pool.starmap_async(average_fes, [(frame, sorted_X_s, sorted_Y_s, sorted_X_l, sorted_Y_l) for frame in range(start,end+1)]).get()
#     pool.close
#     xyz = np.array(xyz)

#     print('collected landscapes')
#     pool = mp.Pool(mp.cpu_count())
#     zboot= pool.map(bootstrap, [xyz[:,2][:,i] for i in range(len(xyz[0][0]))])
#     pool.close
#     print('bootstrapped')
#     zboot=np.array(zboot)
#     print('array')

#     with open('fes_bootstrapped-'+str(start)+'-'+str(end), 'w') as fes_averaged:
#         for line in range(len(xyz[0][0])):
#             fes_averaged.write(str(xyz[0][0][line])+'   '+str(xyz[0][1][line])+'   '+str(zboot[:,0][line])+'   '+str(zboot[:,1][line])+'\n')        

# ### get values from timepoints
# if args.f == 'strip':
#     start, end=args.s,args.e
#     pool = mp.Pool(mp.cpu_count())
#     timecourse = pool.map_async(strip, [row for row in range(start,end+1)]).get()
#     pool.close
#     sorted_timecourse = sorted(timecourse, key = lambda x:(x[0]))
#     with open('energies_time', 'w') as enloc:
#         for line in sorted_timecourse:
#             line_outputs=''
#             for value in line:
#                 line_outputs+=str(value)+'   '
#             enloc.write(line_outputs+'\n')

# if args.f == 'converge':
#     hills_converge()

# if args.f == 'frames':  ## needs updating
#     start, end= args.s, args.e
#     for i in range(start,end):
#         get_frames(i)

# if args.f == '1d_time':
#     if not os.path.exists(param['1d_location']):
#         os.mkdir(param['1d_location'])
#     start, end= args.s, args.e
#     frames=[]
#     for fes_frame in range(start,end+1):
#         if os.path.exists(param['prefix']+str(fes_frame)+'.dat'):
#             cont=False
#             for site in range(1,len(param['ring_location'])+1):
#                 if not os.path.exists(param['1d_location']+'/1D_landscape_site-'+str(site)+'_'+str(fes_frame)):
#                     cont=True
#                     break
#             if cont==True:
#                 frames.append(fes_frame)
#     if len(frames) >= 1: 
#         for frame in frames:
#             average_1d(frame)

# if args.f == 'boot':
#     bootstrap_1d()

# if args.f == 'cv_plot':
#     plot_CV()

# if args.f == 'timecourse':
#     timecourse()
import functions
import argparse
import os, sys
import numpy as np
import multiprocessing as mp

parser = argparse.ArgumentParser()
parser.add_argument('-f', help='function',metavar='plot',type=str, choices= ['sort', 'skip', 'plot', 'boot', 'concat','strip', 'converge', 'frames', '1d_time', 'cv_plot', 'timecourse'], required=True)
parser.add_argument('-input', help='name of input variables',metavar='protein',type=str, required=True)
parser.add_argument('-save', help='save name',metavar='FES',type=str)
parser.add_argument('-d1', help='plots 1d landscape', action='store_true')
parser.add_argument('-bulk', help='plots bulk outline', action='store_true')
parser.add_argument('-error', help='plots error instead of energy', action='store_true')
parser.add_argument('-test', help='runs stripped down version of script', action='store_true')
parser.add_argument('-s', help='start',metavar='1',type=int)
parser.add_argument('-e', help='end',metavar='10',type=int)
args = parser.parse_args()
options = vars(args)

param={'circle_plot':False,'ellipse_plot':False, 'picture':False, '1d':False, 'bulk_outline':False, 'prefix':'fes_', '1d_location':'1d_landscapes', 'invert':1}
circle = ['circle_area','ring_location']
ellipse = ['ellipse_width', 'ellipse_height','ellipse_angle','ring_location']
one_dimension = ['x_points', 'y_points', 'search_width']
picture = ['picture_file','picture_loc']
### working
if args.f == 'sort':
    setting = ['HILLS_sorted', 'HILLS'] 
    param = functions.parameters(args.input, setting, param)
    functions.check_variable(setting, param)
    os.system("awk \'{print $NF,$0}\' "+param['HILLS']+" | sort -n | cut -f2- -d\' \' > "+param['HILLS_sorted'])

if args.f == 'skip':
    setting = ['HILLS_sort', 'HILLS', 'HILLS_skip'] 
    param = functions.parameters(args.input, setting, param)
    functions.check_variable(setting, param)
    if not os.path.exists(param['HILLS_sorted']):
        os.system("awk \'{print $NF,$0}\' "+param['HILLS']+" | sort -n | cut -f2- -d\' \' > "+param['HILLS_sorted'])
    functions.skip_hills(param['HILLS_sorted'], param['HILLS_skip'])

if args.f == 'plot':
    setting = ['picture', 'bulk_values', 'fes', 'energy_max', 'invert','colour_bar_tick', 'CV1','CV2', 'interval_x', 'interval_y', 'bulk_outline', 
               'circle_plot','ellipse_plot', '1d', 'step'] 
    param = functions.parameters(args.input, setting, param)
    functions.check_variable(setting, param)
    if param['circle_plot']:
        param = functions.parameters(args.input, circle, param)
        functions.check_variable(circle, param)
    if param['ellipse_plot']:
        param = functions.parameters(args.input, ellipse, param)
        functions.check_variable(ellipse, param)
    if param['1d']:
        param = functions.parameters(args.input, one_dimension, param)
        functions.check_variable(one_dimension, param)
    if param['picture']:
        param = functions.parameters(args.input, picture, param)
        functions.check_variable(picture, param)
    functions.final_frame(param, args.error, args.save)

#### average z across frames
if args.f == 'concat':
    setting = ['bulk_values', 'prefix', 'start', 'end'] 
    param = functions.parameters(args.input, setting, {})    
    if os.path.exists(param['bulk_values']):
        coord=np.genfromtxt(param['bulk_values'], autostrip=True)
        sorted_X_s, sorted_Y_s, sorted_X_l, sorted_Y_l=coord[:,0],coord[:,1],coord[:,2],coord[:,3]
    else:
        x,y,z,e=functions.readfes(param['prefix']+str(param['end'])+'.dat')
        bulk, sorted_X_s, sorted_Y_s, sorted_X_l, sorted_Y_l = functions.bulk_val(x,y,z)

    pool = mp.Pool(mp.cpu_count())

    xyz = pool.starmap_async(functions.average_fes, [(param, frame, sorted_X_s, sorted_Y_s, sorted_X_l, sorted_Y_l) for frame in range(param['start'],param['end']+1)]).get()
    pool.close
    xyz = np.array(xyz)

    print('collected landscapes')
    pool = mp.Pool(mp.cpu_count())
    zboot= pool.map(functions.bootstrap, [xyz[:,2][:,i] for i in range(len(xyz[0][0]))])
    pool.close
    print('bootstrapped')
    zboot=np.array(zboot)
    print('array')

    with open('fes_bootstrapped-'+str(param['start'])+'-'+str(param['end']), 'w') as fes_averaged:
        for line in range(len(xyz[0][0])):
            fes_averaged.write(str(xyz[0][0][line])+'   '+str(xyz[0][1][line])+'   '+str(zboot[:,0][line])+'   '+str(zboot[:,1][line])+'\n')        

### get values from timepoints
if args.f == 'strip':
    pool = mp.Pool(mp.cpu_count())
    timecourse = pool.starmap_async(functions.strip, [(param, frame)  for frame in range(param['start'],param['end']+1)]).get()
    pool.close
    sorted_timecourse = sorted(timecourse, key = lambda x:(x[0]))
    with open('energies_time', 'w') as enloc:
        for line in sorted_timecourse:
            line_outputs=''
            for value in line:
                line_outputs+=str(value)+'   '
            enloc.write(line_outputs+'\n')

if args.f == 'converge':
    setting = ['bulk_values', 'HILLS_skipped', 'ring_location', 'start', 'end'] 
    param = functions.parameters(args.input, setting, param) 
    if param['circle_plot']:
        param = functions.parameters(args.input, circle, param)
    if param['ellipse_plot']:
        param = functions.parameters(args.input, ellipse, param)
    functions.hills_converge(param)

if args.f == 'frames':  ## needs updating
    for i in range(param['start'],param['end']+1):
        functions.get_frames(i)

if args.f == '1d_time':
    if not os.path.exists(param['1d_location']):
        os.mkdir(param['1d_location'])
    frames=[]
    for fes_frame in range(param['start'],param['end']+1):
        if os.path.exists(param['prefix']+str(fes_frame)+'.dat'):
            cont=False
            for site in range(1,len(param['ring_location'])+1):
                if not os.path.exists(param['1d_location']+'/1D_landscape_site-'+str(site)+'_'+str(fes_frame)):
                    cont=True
                    break
            if cont==True:
                frames.append(fes_frame)
    if len(frames) >= 1: 
        for frame in frames:
            functions.average_1d(param, frame)

if args.f == 'boot':
    functions.bootstrap_1d(param)

if args.f == 'cv_plot':
    functions.plot_CV(param)

if args.f == 'timecourse':
    functions.timecourse(param)
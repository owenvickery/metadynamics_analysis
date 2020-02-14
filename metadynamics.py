import functions
import argparse
import os, sys
import numpy as np
import multiprocessing as mp

parser = argparse.ArgumentParser()
parser.add_argument('-f', help='function',metavar='plot',type=str, choices= ['sort', 'skip', 'plot', 'boot', 'concat','strip', 'converge', 'frames', '1d_time', 'cv_plot', 'timecourse'], required=True)
parser.add_argument('-input', help='name of input variables',metavar='protein',type=str, required=True)
parser.add_argument('-show', help='show interactive plots', action='store_true')
parser.add_argument('-error', help='plots error instead of energy', action='store_true')
parser.add_argument('-info', help='print script information', action='store_true')
args = parser.parse_args()
options = vars(args)

param={'circle_plot':False,'ellipse_plot':False, 'picture':False, '1d':False, 'bulk_outline':False, 'prefix':'fes_', '1d_location':'1d_landscapes'}
circle = ['circle_area','circle_centers']
ellipse = ['ellipse_width', 'ellipse_height','ellipse_angle','ellipse_centers']
one_dimension = ['start_points', 'end_points', 'search_width']
picture = ['picture_file','picture_loc']
### working

if args.info:
    functions.info()

if args.f == 'sort':
    setting = ['HILLS_sort', 'HILLS'] 
    param = functions.parameters(args.input, setting, param)
    functions.check_variable(setting, param)
    if not os.path.exists(param['HILLS_sort']):
        os.system("awk \'{print $NF,$0}\' "+param['HILLS']+" | sort -n | cut -f2- -d\' \' > "+param['HILLS_sort'])
    else:
        sys.exit('The file : '+param['HILLS_sort']+' already exists!')
if args.f == 'skip':
    setting = ['HILLS_sort', 'HILLS', 'HILLS_skip'] 
    param = functions.parameters(args.input, setting, param)
    functions.check_variable(setting, param)
    if not os.path.exists(param['HILLS_sort']):
        os.system("awk \'{print $NF,$0}\' "+param['HILLS']+" | sort -n | cut -f2- -d\' \' > "+param['HILLS_sort'])
    functions.skip_hills(param['HILLS_sort'], param['HILLS_skip'])

if args.f == 'plot':
    setting = ['picture', 'bulk_values', 'fes', 'plot_energy_max','plot_colour_bar_tick', 'CV1','CV2', 'plot_interval_x', 'plot_interval_y', 'bulk_outline', 
               'circle_plot','ellipse_plot', '1d', 'plot_step'] 
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
    functions.final_frame(param, args.error, args.show)

#### average z across frames
if args.f == 'concat':
    setting = ['bulk_values', 'prefix', 'start', 'end'] 
    param = functions.parameters(args.input, setting, param)    
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
    setting = ['circle_plot','ellipse_plot','bulk_values', 'prefix', 'start', 'end'] 
    param = functions.parameters(args.input, setting, param)  
    if param['circle_plot']:
        param = functions.parameters(args.input, circle, param)
        functions.check_variable(circle, param)
    if param['ellipse_plot']:
        param = functions.parameters(args.input, ellipse, param)
        functions.check_variable(ellipse, param)
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
    setting = ['bulk_values', 'HILLS_skip', 'start', 'end', 'converge_labels', 'converge_title_height',
                'labels_size', 'circle_plot', 'ellipse_plot', 'converge_x_interval'] 
    param = functions.parameters(args.input, setting, param) 
    if param['circle_plot']:
        param = functions.parameters(args.input, circle, param)
    if param['ellipse_plot']:
        param = functions.parameters(args.input, ellipse, param)
    functions.hills_converge(param, args.show)

if args.f == 'frames':  ## needs updating
    for i in range(param['start'],param['end']+1):
        functions.get_frames(i)

if args.f == '1d_time':
    setting = ['start', 'end']+one_dimension 
    param = functions.parameters(args.input, setting, param)
    if not os.path.exists(param['1d_location']):
        os.mkdir(param['1d_location'])
    frames=[]
    for fes_frame in range(param['start'],param['end']+1):
        if os.path.exists(param['prefix']+str(fes_frame)+'.dat'):
            cont=False
            for site in range(1,len(param['start_points'])+1):
                if not os.path.exists(param['1d_location']+'/1D_landscape_site-'+str(site)+'_'+str(fes_frame)):
                    cont=True
                    break
            if cont==True:
                frames.append(fes_frame)
    if len(frames) >= 1: 
        for frame in frames:
            functions.average_1d(param, frame)

if args.f == 'boot':
    setting = ['start', 'end', 'equilibration', 'stride', 'ref_min','ref_max', 'min_cv', 'cv_max', 
                'boot_title_height', 'boot_label_y', 'boot_energy_max', 'boot_step_yaxis', 
                'boot_labels_size']+one_dimension 
    param = functions.parameters(args.input, setting, param)
    functions.bootstrap_1d(param, args.show)

if args.f == 'cv_plot':
    setting = ['walker_range','cv_title_height','cv_tick_size', 'cv_labels_size', 'circle_plot', 
                'ellipse_plot', 'CV1', 'CV2', 'cv_x_interval', 'cv_y_interval']
    param = functions.parameters(args.input, setting, param)
    if param['circle_plot']:
        param = functions.parameters(args.input, circle, param)
    if param['ellipse_plot']:
        param = functions.parameters(args.input, ellipse, param)
    functions.plot_CV(param, args.show)

if args.f == 'timecourse':
    setting = ['cv_min', 'cv_max', 'step_xaxis', 'step_yaxis']+one_dimension
    param = functions.parameters(args.input, setting, param)
    functions.timecourse(param, args.show)
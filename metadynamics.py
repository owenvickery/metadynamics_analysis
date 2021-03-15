import functions
import argparse
import os, sys
import numpy as np
import multiprocessing as mp

parser = argparse.ArgumentParser()
parser.add_argument('-f', help='function',metavar='plot',type=str, choices= ['pdb', 'sort', 'skip', 'plot', 'converge', 'boot', 'concat','strip', 'frames', '1d_time', 'cv_plot', 'timecourse'], required=True)
parser.add_argument('-input', help='name of input variables',metavar='protein',type=str)#, required=True)
parser.add_argument('-show', help='show interactive plots', action='store_true')
parser.add_argument('-error', help='plots error instead of energy', action='store_true')
parser.add_argument('-info', help='print script information', action='store_true')
args = parser.parse_args()
options = vars(args)

param_switches={'circle_plot':False,'ellipse_plot':False, 'picture':False, '1d':False, 'bulk_outline':False, 'hills_trim':False}

param_initial_values = {'bulk_outline_shrink':0.5, 'bulk_area':0.5, 'CV1' : 'CV 1','CV2': 'CV 2', 'trim':np.array([]), 'invert_x':1, 'invert_y':1}

param_plot_default = {'plot_colour_bar_tick' : 10, 'plot_energy_step' : 5, 'plot_error_step' : 5, 'plot_energy_max' : False, 
                        'plot_energy_min' : False, 'plot_error_max' : False, 'plot_error_min' : False, 'plot_interval_x' : 2, 
                        'plot_interval_y' : 2, 'plot_trim':np.array([])}

param_input_names={'prefix':'fes_', 'fes':'fes.dat', '1d_location':'1d_landscapes', 'bulk_values':'bulk_values', 
                  'HILLS_skip':'HILLS_skipped', 'HILLS_sort':'HILLS_sorted','HILLS':'HILLS'}   

param_pdb = {'pdb_output':'fes_pdb.pdb', 'pdb_offset':np.array([0,0,0])}   

param_converge = {'converge_x_interval':False, 'converge_y_interval':False, 'converge_labels':35, 'converge_title_height':0.8}

param = {**param_switches, **param_initial_values, **param_input_names, **param_pdb, **param_plot_default, **param_converge}

circle = ['circle_area','circle_centers']

ellipse = ['ellipse_width', 'ellipse_height','ellipse_angle','ellipse_centers']

one_dimension = ['start_points', 'end_points', 'search_width']

picture = ['picture_file','picture_loc']
### working

if args.info:
    functions.info()

if args.f == 'pdb':
    setting = ['bulk_values', 'fes', 'bulk_outline', 'plot_trim','trim', 'bulk_outline_shrink', 'bulk_area', 
               'plot_energy_max','pdb_offset', 'pdb_output', 'invert_x', 'invert_y'] 
    param = functions.parameters(args.input, setting, param)
    functions.check_variable(setting, param)
    functions.plot_pdb(param)

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
    setting = ['picture', 'bulk_values', 'fes','plot_colour_bar_tick', 'CV1','CV2', 'plot_interval_x',
               'plot_interval_y', 'bulk_outline', 'circle_plot','ellipse_plot', '1d', 'plot_energy_step', 
               'plot_error_step', 'trim', 'bulk_outline_shrink', 'bulk_area', 'plot_trim', 'plot_error_max', 'plot_error_min',
               'plot_energy_max','plot_energy_min', 'invert_x', 'invert_y']
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
    X,Y,Z,E, bulk = functions.final_frame(param, args.error, args.show)
    if args.error:
        functions.plot_error(X,Y,Z,E,bulk, param, args.show)

#### average z across frames
if args.f == 'concat':
    print('Averaging and getting error from FES')
    setting = ['bulk_values', 'prefix', 'frame_start', 'frame_end', 'trim', 'invert_x', 'invert_y'] 
    param = functions.parameters(args.input, setting, param)    
    functions.check_variable(setting, param)
    if os.path.exists(param['bulk_values']):
        coord=np.genfromtxt(param['bulk_values'], autostrip=True)
        sorted_X_s, sorted_Y_s, sorted_X_l, sorted_Y_l=coord[:,0],coord[:,1],coord[:,2],coord[:,3]
    else:
        x,y,z,e=functions.readfes(param['prefix']+str(param['end'])+'.dat')
        bulk, sorted_X_s, sorted_Y_s, sorted_X_l, sorted_Y_l = functions.bulk_val(x,y,z)

    pool = mp.Pool(8)
    xyz = pool.starmap_async(functions.average_fes, [(param, frame, sorted_X_s, sorted_Y_s, sorted_X_l, sorted_Y_l) for frame in range(param['start'],param['end']+1)]).get()
    pool.close
    xyz = np.array(xyz)

    print('collected landscapes')
    pool = mp.Pool(8)
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
    setting = ['circle_plot','ellipse_plot','bulk_values', 'prefix', 'frame_start', 'frame_end', 'invert_x', 'invert_y'] 
    param = functions.parameters(args.input, setting, param)  
    if param['circle_plot']:
        param = functions.parameters(args.input, circle, param)
        setting+=circle
    if param['ellipse_plot']:
        param = functions.parameters(args.input, ellipse, param)
        setting+=ellipse
    functions.check_variable(setting, param)
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
    setting = ['bulk_values', 'HILLS_skip', 'converge_labels', 'converge_title_height',
                'labels_size', 'circle_plot', 'ellipse_plot', 'converge_x_interval', 'converge_y_interval',
                'hills_trim', 'invert_x', 'invert_y'] 
    param = functions.parameters(args.input, setting, param) 
    if param['circle_plot']:
        param = functions.parameters(args.input, circle, param)
    if param['ellipse_plot']:
        param = functions.parameters(args.input, ellipse, param)
    functions.hills_converge(param, args.show)

if args.f == 'frames':  ## needs updating
    setting = ['bulk_values', 'HILLS_skip', 'start', 'end', 'converge_labels', 'converge_title_height',
                'labels_size', 'circle_plot', 'ellipse_plot', 'converge_x_interval'] 
    param = functions.parameters(args.input, setting, param) 
    if param['circle_plot']:
        param = functions.parameters(args.input, circle, param)
    if param['ellipse_plot']:
        param = functions.parameters(args.input, ellipse, param)
    # for i in range(param['start'],param['end']+1):
    functions.get_frames(param)

if args.f == '1d_time':
    setting = ['frame_start', 'frame_end',  'invert_x', 'invert_y']+one_dimension 
    param = functions.parameters(args.input, setting, param)
    if not os.path.exists(param['1d_location']):
        os.mkdir(param['1d_location'])
    frames=[]
    for fes_frame in range(param['frame_start'],param['frame_end']+1):
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
    setting = ['frame_start', 'frame_end',  'equilibration', 'stride', 'ref_min','ref_max', 'min_cv', 'cv_max', 
                'boot_title_height', 'boot_label_y', 'boot_energy_max', 'boot_step_yaxis', 
                'boot_labels_size', 'invert_x', 'invert_y']+one_dimension 
    param = functions.parameters(args.input, setting, param)
    functions.bootstrap_1d(param, args.show)

if args.f == 'cv_plot':
    setting = ['walker_range','cv_title_height','cv_tick_size', 'cv_labels_size', 'circle_plot', 
                'ellipse_plot', 'CV1', 'CV2', 'cv_x_interval', 'cv_y_interval', 'invert_x', 'invert_y']
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
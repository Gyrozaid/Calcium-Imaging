
# imports
import bokeh.plotting as bpl
import cv2
import glob
import holoviews as hv
from IPython import get_ipython
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import psutil
import seaborn as sns
from pprint import pp
from tqdm import tqdm

import caiman as cm
from caiman.source_extraction import cnmf
from caiman.source_extraction.cnmf.cnmf import load_CNMF
from caiman.utils.utils import download_demo
from caiman.utils.visualization import inspect_correlation_pnr, nb_inspect_correlation_pnr
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import params as params
from caiman.utils.visualization import plot_contours, nb_view_patches, nb_plot_contour
from caiman.utils.visualization import view_quilt


# set env variables in case they weren't already set
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

#collect all recording paths
project_path = r"C:\Users\Moorman\calcium_imaging\raw\7Trial_Habituation_Dishabituation_Task_Final_Recording_Jul232025\IL16-24-086\2025_07_23"
recording_paths = {}

for trial in os.listdir(project_path):
    trial_path = os.path.join(project_path, trial)
    for folder in os.listdir(trial_path):
        folder_path = os.path.join(trial_path, folder)
        if folder == 'V4_Miniscope':
            recordings = []
            for recording in os.listdir(folder_path):
                if recording.endswith('.avi'):
                    recording_path = os.path.join(folder_path, recording)
                    recordings.append(recording_path)
                        
            recording_paths[trial] = recordings





# MOCO parameters
frate = 15                       # movie frame rate
decay_time = 0.4                 # length of a typical transient in seconds
motion_correct = False    # flag for performing motion correction
pw_rigid = False         # flag for performing piecewise-rigid motion correction (otherwise just rigid)
gSig_filt = (20, 20)       # sigma for high pass spatial filter applied before motion correction, used in 1p data
max_shifts = (6, 6)      # maximum allowed rigid shift
strides = (48, 48)       # start a new patch for pw-rigid motion correction every x pixels
overlaps = (24, 24)      # overlap between patches (size of patch = strides + overlaps)
max_deviation_rigid = 3  # maximum deviation allowed for patch with respect to rigid shifts
border_nan = 'copy'      # replicate values along the boundaries



# CNMFE PARAMS
p = 1               # order of the autoregressive system
K = None            # upper bound on number of components per patch, in general None for CNMFE
gSig = np.array([3, 3])  # expected half-width of neurons in pixels 
gSiz = 2*gSig + 1     # half-width of bounding box created around neurons during initialization
merge_thr = .7      # merging threshold, max correlation allowed
rf = 50             # half-size of the patches in pixels. e.g., if rf=40, patches are 80x80
stride_cnmf = 25    # amount of overlap between the patches in pixels 
tsub = 2            # downsampling factor in time for initialization, increase if you have memory problems
ssub = 1            # downsampling factor in space for initialization, increase if you have memory problems
gnb = 0             # number of background components (rank) if positive, set to 0 for CNMFE
low_rank_background = None  # None leaves background of each patch intact (use True if gnb>0)
nb_patch = 0        # number of background components (rank) per patch (0 for CNMFE)
min_corr = .5       # min peak value from correlation image
min_pnr = 5        # min peak to noise ration from PNR image
ssub_B = 2          # additional downsampling factor in space for background (increase to 2 if slow)
ring_size_factor = 1.4  # radius of ring is gSiz*ring_size_factor

#save param
save_results = True



#process
for trial in recording_paths:
    print(F'RUNNING TRIAL {trial}\n')
    movie_paths = recording_paths[trial]

    #skip empty dirs
    if len(movie_paths) == 0:
        continue

    #load movie
    print('LOADING MOVIE\n')
    movie_orig = cm.load_movie_chain(movie_paths)

    #set up processing cluster
    if 'cluster' in locals():  # 'locals' contains list of current local variables
        print('Closing previous cluster\n')
        cm.stop_server(dview=cluster)
    print("Setting up new cluster\n")
    _, cluster, n_processes = cm.cluster.setup_cluster(backend='multiprocessing', 
                                                    n_processes=None, 
                                                    ignore_preexisting=False)
    print(f"Successfully set up cluster with {n_processes} processes")


    #motion correct
    mc_dict = {
        'fnames': movie_paths,
        'fr': frate,
        'decay_time': decay_time,
        'pw_rigid': pw_rigid,
        'max_shifts': max_shifts,
        'gSig_filt': gSig_filt,
        'strides': strides,
        'overlaps': overlaps,
        'max_deviation_rigid': max_deviation_rigid,
        'border_nan': border_nan
    
    }

    parameters = params.CNMFParams(params_dict=mc_dict)


    if motion_correct:
        # do motion correction rigid
        print('MOTION CORRECTING\n')
        mot_correct = MotionCorrect(movie_paths, dview=cluster, **parameters.get_group('motion'))
        mot_correct.motion_correct(save_movie=True)
        fname_mc = mot_correct.fname_tot_els if pw_rigid else mot_correct.fname_tot_rig
        if pw_rigid:
            bord_px = np.ceil(np.maximum(np.max(np.abs(mot_correct.x_shifts_els)),
                                        np.max(np.abs(mot_correct.y_shifts_els)))).astype(int)
        else:
            bord_px = np.ceil(np.max(np.abs(mot_correct.shifts_rig))).astype(int)


        bord_px = 0 if border_nan == 'copy' else bord_px
        fname_new = cm.save_memmap(fname_mc, base_name='memmap_', order='C',
                                border_to_0=bord_px)
    else:  # if no motion correction just memory map the file
        print('NO MOTION CORRECT, SAVING MEMORY MAPPED FILE\n')
        fname_new = cm.save_memmap(movie_paths, base_name='memmap_',
                                order='C', border_to_0=0, dview=cluster)
        

    print('LOADING MEMORY MAPPED FILE\n')
    # load memory mappable file
    Yr, dims, T = cm.load_memmap(fname_new)
    images = Yr.T.reshape((T,) + dims, order='F')



    # CNMFE ALG

    #set params
    parameters.change_params(params_dict={'method_init': 'corr_pnr',  # use this for 1 photon
                                    'K': K,
                                    'gSig': gSig,
                                    'gSiz': gSiz,
                                    'merge_thr': merge_thr,
                                    'p': p,
                                    'tsub': tsub,
                                    'ssub': ssub,
                                    'rf': rf,
                                    'stride': stride_cnmf,
                                    'only_init': True,    # set it to True to run CNMF-E
                                    'nb': gnb,
                                    'nb_patch': nb_patch,
                                    'method_deconvolution': 'oasis',       # could use 'cvxpy' alternatively
                                    'low_rank_background': low_rank_background,
                                    'update_background_components': True,  # sometimes setting to False improve the results
                                    'min_corr': min_corr,
                                    'min_pnr': min_pnr,
                                    'normalize_init': False,               # just leave as is
                                    'center_psf': True,                    # True for 1p
                                    'ssub_B': ssub_B,
                                    'ring_size_factor': ring_size_factor,
                                    'del_duplicates': True,                # whether to remove duplicates from initialization
                                    #'border_pix': bord_px
                                    });                # number of pixels to not consider in the borders)

    print("FITTING CNMFE\n")

    #define model
    cnmfe_model = cnmf.CNMF(n_processes=n_processes, 
                        dview=cluster, 
                        params=parameters)
    
    #extract correlation image
    gsig_tmp = (3,3)
    correlation_image, peak_to_noise_ratio = cm.summary_images.correlation_pnr(images[::max(T//1000, 1)], # subsample if needed
                                                                            gSig=gsig_tmp[0], # used for filter
                                                                            swap_dim=False) # change swap dim if output looks weird, it is a problem with tiffile
    
    #fit model
    cnmfe_model.fit(images)

    print('SAVING RESULTS\n')
    #save results
    if save_results:


        save_path =  f"C:\\Users\\Moorman\\calcium_imaging\\processed\\IL16-24-086\\2025_07_23\\{trial}"

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        

        cnmfe_model.estimates.Cn = correlation_image # squirrel away correlation image with cnmf object
        cnmfe_model.save(save_path + '\\analyzed_neurons.hdf5')


    #reset cluster
    print("RESET CLUSTER")
    del cnmfe_model
    del movie_orig
    del images
    del Yr
    del mot_correct
    cm.stop_server(dview=cluster)
    cluster=None


    break




    


    

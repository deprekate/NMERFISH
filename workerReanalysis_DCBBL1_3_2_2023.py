
#activate cellpose&&python C:\Scripts\NMERFISH\workerReanalysis_DCBBL1_1year1wkNJ_ASO_SAL_3_31_2023.py

#save_folder =r'\\192.168.0.100\bbfish100\DCBBL1_4week_6_2_2023\MERFISH_Analysis'
#save_folder =r'\\192.168.0.10\bbfishdc13\DCBBL1_3_2_2023\MERFISH_Analysis'
#save_folder =r'\\192.168.0.21\bbfishdc21\DCBBL1_1year1wkNJ_ASO_SAL_3_31_2023\MERFISH_Analysis'
save_folder =r'\\192.168.0.10\bbfishdc13\DCBBL1_3_2_2023\MERFISH_Analysis'
master_analysis_folder = r'C:\Scripts\NMERFISH'
from multiprocessing import Pool, TimeoutError
import time,sys
import os,sys,numpy as np

sys.path.append(master_analysis_folder)
from ioMicro import *


def main_f_compute(ifov):
    fovs_sets_fl = save_folder+os.sep+'fl_fovs_sets.npy'
    if os.path.exists(fovs_sets_fl):
        elems = np.load(fovs_sets_fl)
    else:
        fov_fls = glob.glob(save_folder+os.sep+'fov*')
        elems = [(fov.replace('.zarr',''),fov_fl.split('__')[-1].split('.')[0]) 
                for fov_fl in fov_fls for fov in np.load(fov_fl)]
        np.save(fovs_sets_fl,elems)
    if ifov<len(elems):
        fov,set_ = elems[ifov]

        dec = decoder_simple(save_folder,fov,set_)

        dec.get_XH(dec.fov,dec.set_,ncols=3)#number of colors match 
        dec.XH = dec.XH[dec.XH[:,-4]>0.25] ### keep the spots that are correlated with the expected PSF for 60X
        ### compute better drift
        dec.XH_save = dec.XH.copy()
        #dec.XH = dec.XH[dec.XH[:,-3]>3000] ### keep the spots that are correlated with the expected PSF for 60X
        dec.load_library(lib_fl = r'\\192.168.0.10\bbfishdc13\codebook_0_New_DCBB-300_MERFISH_encoding_2_21_2023.csv',nblanks=-1)
        keep_best_N_for_each_Readout(dec,Nkeep = 15000)
        dec.get_inters(dinstance_th=5,enforce_color=True)# enforce_color=False
        dec.get_icodes(nmin_bits=4,method = 'top4',norm_brightness=-1)
        apply_fine_drift(dec,plt_val=True)
        dec.XH = dec.XH_save.copy()
        R = dec.XH[:,-1].astype(int)
        dec.XH[:,:3] -= dec.drift_arr[R]

        dec.get_inters(dinstance_th=2,enforce_color=True)# enforce_color=False
        dec.get_icodes(nmin_bits=4,method = 'top4',norm_brightness=-1)

        #apply_flat_field(dec,tag='med_col')
        #apply_fine_drift(dec,plt_val=True)
        #scoresRefT = get_score_per_color(dec)
        #get_score_withRef(dec,scoresRefT,plt_val=True,gene=None,iSs = None)
        #dec.th=-1.5
        #plot_statistics(dec)
    return ifov

def main_f(ifov,force=False,try_mode=True):
    fl_tag = save_folder+os.sep+str(ifov).zfill(5)+'_recomputed.txt'
    if not os.path.exists(fl_tag) or force:
        if try_mode:
            try:
                main_f_compute(ifov)
                fid  = open(fl_tag,'w')
                fid.close()
            except:
                print("Failed",ifov)
        else:
            main_f_compute(ifov)
            fid  = open(fl_tag,'w')
            fid.close()
    return ifov
    
    
if __name__ == '__main__':

    items = list(range(1500))[::-1]
    main_f(615,force=False,try_mode=False)
    if True:
        with Pool(processes=10) as pool:
            print('starting pool')
            result = pool.map(main_f, items)

    
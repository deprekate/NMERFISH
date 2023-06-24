
#activate cellpose&&python C:\Scripts\NMERFISH\workerSaveFinal2.py

save_folder =r'\\192.168.0.100\bbfish100\DCBBL1_1year1wkNJ_ASO_SAL_4_18_2023\MERFISH_Analysis'

master_analysis_folder = r'C:\Scripts\NMERFISH'
from multiprocessing import Pool, TimeoutError
import time,sys
import os,sys,numpy as np

sys.path.append(master_analysis_folder)
from ioMicro import *


def main_f(ifov,force=False):
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
        scoresRef = np.load(save_folder+os.sep+'scoresRef.npy',allow_pickle=True)
        #save_final_decoding(save_folder,fov,set_,scoresRef,th=-1.5,plt_val=False,tag_save = 'finaldecs_',apply_flat=False,ncols=3,
        #                            tags_smFISH=['Aldh','Sox11'],
        #                            genes_smFISH=[['Igfbpl1','Aldh1l1','Ptbp1'],['Sox11','Sox2','Dcx']],Hths=[10000,10000,8500],force=force)
        save_final_decoding(save_folder,fov,set_,scoresRef,th=-1.5,plt_val=False,tag_save = 'finaldecs_',apply_flat=True,ncols=3,
                            tags_smFISH=['Aldh','Sox11'],
                            genes_smFISH=[['Igfbpl1','Aldh1l1','Ptbp1'],['Sox11','Sox2','Dcx']],Hths=[3000,2500,2000],force=False)
    return ifov
    
    
    
if __name__ == '__main__':

    items = list(range(1300))
    main_f(1000)
    if True:
        with Pool(processes=10) as pool:
            print('starting pool')
            result = pool.map(main_f, items)

    
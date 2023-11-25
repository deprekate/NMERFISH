
#activate napari&&python "C:\Scripts\NMERFISH\RoyChimera\worker_Scope2__RoyEdu_C_ASO_4week__1mopInj_11_21_2023.py"

master_analysis_folder = r'C:\Scripts\NMERFISH' ## where the code libes
lib_fl = r'C:\Scripts\NMERFISH\codebooks\codebook_0_New_DCBB-300_MERFISH_encoding_2_21_2023.csv' ### codebook
### Did you compute PSF and median flat field images?
psf_file = r'C:\Scripts\NMERFISH\psfs\dic_psfff_scope2_40X_750_300pix.pkl'
master_data_folders = [r'\\192.168.0.34\Papaya6\RoyEdu_C_ASO_4week__1mopInj_11_21_2023']
save_folder =r'\\192.168.0.34\Papaya6\RoyEdu_C_ASO_4week__1mopInj_11_21_2023\MERFISH_Analysis'  ###change
flat_field_tag = r'C:\Scripts\NMERFISH\flat_field\Scope2_40X_'
iHm=0 #H iHmin -> H iHmax oly keeps folders of the the form H33,H34...
iHM=16

from multiprocessing import Pool, TimeoutError
import time,sys
import os,sys,numpy as np

sys.path.append(master_analysis_folder)
from ioMicro import *

def main_do_compute_fits(save_folder,fld,fov,icol,save_fl,psf,old_method):
    im_ = read_im(fld+os.sep+fov)
    im__ = np.array(im_[icol],dtype=np.float32)
    
    if old_method:
        ### previous method - no deconvolution
        im_n = norm_slice(im__,s=30)
        Xh = get_local_maxfast_tensor(im_n,th_fit=500,im_raw=im__,dic_psf=None,delta=1,delta_fit=3,sigmaZ=1,sigmaXY=1.5,gpu=False)
    else:
        ### new method
        fl_med = flat_field_tag+'med_col_raw'+str(icol)+'.npz'
        if os.path.exists(fl_med):
            im_med = np.array(np.load(fl_med)['im'],dtype=np.float32)
            im_med = cv2.blur(im_med,(20,20))
            im__ = im__/im_med*np.median(im_med)
            print("Correcting flat field...")
        Xh = get_local_max_tile(im__,th=3600,s_ = 300,pad=100,psf=psf,plt_val=None,snorm=30,gpu=True,
                                deconv={'method':'wiener','beta':0.0001},
                                delta=1,delta_fit=3,sigmaZ=1,sigmaXY=1.5)
    np.savez_compressed(save_fl,Xh=Xh)
def compute_fits(save_folder,fov,all_flds,redo=False,ncols=4,
                psf_file = psf_file,try_mode=True,old_method=False):
    psf = np.load(psf_file,allow_pickle=True)
    
    for fld in tqdm(all_flds):
        for icol in range(ncols-1):
            tag = os.path.basename(fld)
            save_fl = save_folder+os.sep+fov.split('.')[0]+'--'+tag+'--col'+str(icol)+'__Xhfits.npz'
            try:
                np.load(save_fl)
            except:
                redo=True
            if not os.path.exists(save_fl) or redo:
                if try_mode:
                    try:
                        main_do_compute_fits(save_folder,fld,fov,icol,save_fl,psf,old_method)
                    except:
                        print("Failed",fld,fov,icol)
                else:
                    main_do_compute_fits(save_folder,fld,fov,icol,save_fl,psf,old_method)                   
def compute_decoding(save_folder,fov,set_,redo=False):
    dec = decoder_simple(save_folder,fov,set_)
    #self.decoded_fl = self.decoded_fl.replace('decoded_','decodedNew_')
    complete = dec.check_is_complete()
    if complete==0 or redo:
        #compute_drift(save_folder,fov,all_flds,set_,redo=False,gpu=False)
        dec = decoder_simple(save_folder,fov=fov,set_=set_)
        dec.get_XH(fov,set_,ncols=3,nbits=16,th_h=5000)#number of colors match ######################################################## Could change for speed.
        dec.XH = dec.XH[dec.XH[:,-4]>0.25] ### keep the spots that are correlated with the expected PSF for 60X
        dec.load_library(lib_fl,nblanks=-1)
        
        dec.ncols = 3
        if False:
            dec.XH_save = dec.XH.copy()
            keep_best_N_for_each_Readout(dec,Nkeep = 15000)
            dec.get_inters(dinstance_th=5,enforce_color=True)# enforce_color=False
            dec.get_icodes(nmin_bits=4,method = 'top4',norm_brightness=-1)
            apply_fine_drift(dec,plt_val=True)
            dec.XH = dec.XH_save.copy()
            R = dec.XH[:,-1].astype(int)
            dec.XH[:,:3] -= dec.drift_arr[R]
        dec.get_inters(dinstance_th=2,nmin_bits=4,enforce_color=True,redo=True)
        #dec.get_icodes(nmin_bits=4,method = 'top4',norm_brightness=None,nbits=24)#,is_unique=False)
        get_icodesV2(dec,nmin_bits=4,delta_bits=None,iH=-3,redo=False,norm_brightness=False,nbits=48,is_unique=True)

def get_iH(fld): 
    try:
        return int(os.path.basename(fld).split('_')[0][1:])
    except:
        return np.inf
def get_files(set_ifov,iHm=iHm,iHM=iHM,remap=False):
    
    
    if not os.path.exists(save_folder): os.makedirs(save_folder)
    
    #master_folder = master_data_folder
    all_flds = []
    for master_folder in master_data_folders:
        #all_flds = glob.glob(r'\\192.168.0.6\bbfishjoy4\CGBB_embryo_4_28_2023\H*MER_*')
        all_flds += glob.glob(master_folder+r'\H*')
    #all_flds += glob.glob(master_folder+r'\P*')
    
    ### reorder based on hybe
    all_flds = np.array(all_flds)[np.argsort([get_iH(fld)for fld in all_flds])] 
    
    set_,ifov = set_ifov
    all_flds = [fld for fld in all_flds if set_ in os.path.basename(fld)]
    
    all_flds = [fld for fld in all_flds if ((get_iH(fld)>=iHm) and (get_iH(fld)<=iHM))]
    
    fovs_fl = save_folder+os.sep+'fovs__'+set_+'.npy'
    if not os.path.exists(fovs_fl) or remap:
        folder_map_fovs = [fld for fld in all_flds if 'low' not in os.path.basename(fld)][0]
        fls = glob.glob(folder_map_fovs+os.sep+'*.zarr')
        fovs = np.sort([os.path.basename(fl) for fl in fls])
        np.save(fovs_fl,fovs)
    else:
        fovs = np.sort(np.load(fovs_fl))
    fov=None
    if ifov<len(fovs):
        fov = fovs[ifov]
        all_flds = [fld for fld in all_flds if os.path.exists(fld+os.sep+fov)]
    return save_folder,all_flds,fov
        
        


############### New Code inserted here!!! ##################
### First copy ioMicro to the other computer from Scope1A1
### To analyze only D9 change items to ['_D9']
### Move the decodedNew files and the driftNew files to another folder #############!!!!!!
def compute_drift_features(save_folder,fov,all_flds,set_,redo=False,gpu=True):
    fls = [fld+os.sep+fov for fld in all_flds]
    for fl in fls:
        get_dapi_features(fl,save_folder,set_,gpu=gpu,im_med_fl = flat_field_tag+r'med_col_raw3.npz',
                    psf_fl = psf_file,redo=redo)
                    
def get_best_translation_pointsV2(fl,fl_ref,save_folder,set_,resc=5):
    
    obj = get_dapi_features(fl,save_folder,set_)
    obj_ref = get_dapi_features(fl_ref,save_folder,set_)
    tzxyf,tzxy_plus,tzxy_minus,N_plus,N_minus = np.array([0,0,0]),np.array([0,0,0]),np.array([0,0,0]),0,0
    if (len(obj.Xh_plus)>0) and (len(obj_ref.Xh_plus)>0):
        X = obj.Xh_plus[:,:3]
        X_ref = obj_ref.Xh_plus[:,:3]
        tzxy_plus,N_plus = get_best_translation_points(X,X_ref,resc=resc,return_counts=True)
    if (len(obj.Xh_minus)>0) and (len(obj_ref.Xh_minus)>0):
        X = obj.Xh_minus[:,:3]
        X_ref = obj_ref.Xh_minus[:,:3]
        tzxy_minus,N_minus = get_best_translation_points(X,X_ref,resc=resc,return_counts=True)
    if np.max(np.abs(tzxy_minus-tzxy_plus))<=2:
        tzxyf = -(tzxy_plus*N_plus+tzxy_minus*N_minus)/(N_plus+N_minus)
    else:
        tzxyf = -[tzxy_plus,tzxy_minus][np.argmax([N_plus,N_minus])]
    

    return [tzxyf,tzxy_plus,tzxy_minus,N_plus,N_minus]
def compute_drift_V2(save_folder,fov,all_flds,set_,redo=False,gpu=True):
    drift_fl = save_folder+os.sep+'driftNew_'+fov.split('.')[0]+'--'+set_+'.pkl'
    if not os.path.exists(drift_fl) or redo:
        fls = [fld+os.sep+fov for fld in all_flds]
        fl_ref = fls[len(fls)//2]
        newdrifts = []
        for fl in fls:
            drft = get_best_translation_pointsV2(fl,fl_ref,save_folder,set_,resc=5)
            print(drft)
            newdrifts.append(drft)
        pickle.dump([newdrifts,all_flds,fov,fl_ref],open(drift_fl,'wb'))
def compute_main_f(save_folder,all_flds,fov,set_,ifov,redo_fits,redo_drift,redo_decoding,try_mode,old_method):
    print("Computing fitting on: "+str(fov))
    print(len(all_flds),all_flds)
    compute_fits(save_folder,fov,all_flds,redo=redo_fits,try_mode=try_mode,old_method=old_method)
    print("Computing drift on: "+str(fov))
    #compute_drift(save_folder,fov,all_flds,set_,redo=redo_drift)
    compute_drift_features(save_folder,fov,all_flds,set_,redo=redo_drift,gpu=True)
    #compute_drift_V2(save_folder,fov,all_flds,set_,redo=redo_drift,gpu=True)
    #compute_decoding(save_folder,fov,set_,redo=redo_decoding)

    
############### End Code inserted here!!! ##################

def main_f(set_ifov,redo_fits = False,redo_drift=False,redo_decoding=False,try_mode=True,old_method=False):
    set_,ifov = set_ifov
    save_folder,all_flds,fov = get_files(set_ifov)
    
    if try_mode:
        try:
            compute_main_f(save_folder,all_flds,fov,set_,ifov,redo_fits,redo_drift,redo_decoding,try_mode,old_method)
        except:
            print("Failed within the main analysis:")
    else:
        compute_main_f(save_folder,all_flds,fov,set_,ifov,redo_fits,redo_drift,redo_decoding,try_mode,old_method)
    
    return set_ifov
    
def cleanup_decoded_drift(save_folder):
    fls = glob.glob(save_folder+os.sep+'decoded*')
    fls+= glob.glob(save_folder+os.sep+'drift*')
    for fl in tqdm(fls):
        os.remove(fl)
    
if __name__ == '__main__':
    # start 4 worker processes
    items = [(set_,ifov) for set_ in ['_set1','_set2','_set3']
                        for ifov in range(450)]
    
    inds = np.random.permutation(np.arange(len(items)))###
    items = [items[i] for i in inds]####
    #for set_ in ['_D7','_D9','_D13','_D14','_D16']:
    #    get_files([set_,0],remap=True)
    #main_f(('_D16', 211),try_mode=False)#
    #for item in [['_D16',211],['_D16',169]][::-1]:
       
    
    main_f(('_set2',78),redo_drift=False,redo_decoding=False,try_mode=False)
    if True:
        with Pool(processes=3) as pool:
            print('starting pool')
            result = pool.map(main_f, items)
    #activate napari&&python "C:\Scripts\NMERFISH\RoyChimera\worker_Scope2__RoyEdu_C_ASO_4week__1mopInj_11_21_2023.py"
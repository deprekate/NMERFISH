#activate napari&&python "C:\Scripts\NMERFISH\worker_Scope1__EmbryoPlateE1.py"
from multiprocessing import Pool, TimeoutError
import time,sys
import os,sys,numpy as np

master_analysis_folder = r'C:\Scripts\NMERFISH'
library_file = master_analysis_folder+r'\codebooks\blank_codebook_Mahsa_DevEP1-code_color2__comb16-4-4.csv'
psf_file = master_analysis_folder+r'\psfs\psf_750_Scope1_embryo_big_final.npy'
sys.path.append(master_analysis_folder)
from ioMicro import *


def compute_drift(save_folder,fov,all_flds,set_,redo=False,gpu=False,szz = 25):
    """
    save_folder where to save to - analysis_fodler
    fov - i.e. Conv_zscan_005.zarr
    all_flds - folders that contain eithger the MERFISH bits or control bits or sm
    
    """
    #print(len(all_flds))
    #print(all_flds)
    drift_fl = save_folder+os.sep+'drift_'+fov.split('.')[0]+'--'+set_+'.pkl'
    iiref = None
    previous_drift = {}
    if not os.path.exists(drift_fl) or redo:
        redo = True
    else:
        drifts_,all_flds_,fov_ = pickle.load(open(drift_fl,'rb'))
        all_tags_ = np.array([os.path.basename(fld)for fld in all_flds_])
        all_tags = np.array([os.path.basename(fld)for fld in all_flds])
        iiref = np.argmin([np.sum(np.abs(drift[0]))for drift in drifts_])
        previous_drift = {tag:drift for drift,tag in zip(drifts_,all_tags_)}
        if not (len(all_tags_)==len(all_tags)):
            redo = True
        else:
            if not np.all(np.sort(all_tags_)==np.sort(all_tags)):
                redo = True
    if redo:
        print("Computing drift...")
        ims = [read_im(fld+os.sep+fov) for fld in all_flds] #map the image
        ncols,sz,sx,sy = ims[0].shape
        sls = slice((sz-szz)//2,(sz+szz)//2)
        if iiref is None: iiref = len(ims)//2
        im_ref = np.array(ims[iiref][-1][sls],dtype=np.float32)
        all_tags = np.array([os.path.basename(fld)for fld in all_flds])
        drifts = [previous_drift.get(tag,get_txyz(im[-1][sls],im_ref,sz_norm=30, sz=600,gpu=gpu)) 
                    for im,tag in zip(tqdm(ims),all_tags)]
        
        pickle.dump([drifts,all_flds,fov],open(drift_fl,'wb'))
def compute_fits(save_folder,fov,all_flds,redo=False,ncols=4):
    for fld in tqdm(all_flds):
        
        #ncols = len(im_)
        for icol in range(ncols-1):
            #icol=2
            tag = os.path.basename(fld)
            save_fl = save_folder+os.sep+fov.split('.')[0]+'--'+tag+'--col'+str(icol)+'__Xhfits.npz'
            if not os.path.exists(save_fl) or redo:
                try:
                    im_ = read_im(fld+os.sep+fov)
                    im__ = np.array(im_[icol],dtype=np.float32)
                    
                    #print("Reading image")
                    if False:
                        ### previous method
                        im_n = norm_slice(im__,s=30)
                        #print("Fitting image")
                        Xh = get_local_max(im_n,500,im_raw=im__,dic_psf=None,delta=1,delta_fit=3,dbscan=True,
                              return_centers=False,mins=None,sigmaZ=1,sigmaXY=1.5)
                    else:
                        fl_med = save_folder+os.sep+'med_col_raw'+str(icol)+'.npy'
                        if os.path.exists(fl_med):
                            im_med = np.array(np.load(fl_med)['im'],dtype=np.float32)
                            im_med = cv2.blur(im_med,(20,20))
                            im__ = im__/im_med*np.median(im_med)
                        psf = np.load(psf_file)  
                        try:
                            Xh = get_local_max_tile(im__,th=3600,s_ = 500,pad=100,psf=psf,plt_val=None,snorm=30,gpu=True,
                                                    deconv={'method':'wiener','beta':0.0001},
                                                    delta=1,delta_fit=3,sigmaZ=1,sigmaXY=1.5)
                        except:
                            Xh = get_local_max_tile(im__,th=3600,s_ = 500,pad=100,psf=psf,plt_val=None,snorm=30,gpu=False,
                                                    deconv={'method':'wiener','beta':0.0001},
                                                    delta=1,delta_fit=3,sigmaZ=1,sigmaXY=1.5)
                    np.savez_compressed(save_fl,Xh=Xh)
                except:
                    print("Failed",fld,fov,icol)
def compute_decoding(save_folder,fov,set_,redo=False):
    dec = decoder_simple(save_folder,fov,set_)
    complete = dec.check_is_complete()
    if complete==0 or redo:
        dec.get_XH(fov,set_,ncols=3)#number of colors match 
        dec.XH = dec.XH[dec.XH[:,-4]>0.25] ### keep the spots that are correlated with the expected PSF for 60X
        dec.load_library(library_file,nblanks=-1)
        dec.get_inters(dinstance_th=2,enforce_color=True)# enforce_color=False
        dec.get_icodes(nmin_bits=4,method = 'top4',norm_brightness=-1)
def get_iH(fld): 
    try:
        return int(os.path.basename(fld).split('_')[0][1:])
    except:
        return np.inf
def get_files(set_ifov,iHm=16*4+1,iHM=16*5):
    
    
    save_folder =r'\\192.168.0.6\bbfishjoy4extra\MERFISH_AnalysisPE1'
    if not os.path.exists(save_folder): os.makedirs(save_folder)
    set_,ifov = set_ifov
    
    all_flds = glob.glob(r'\\192.168.0.6\bbfishjoy4\CGBB_embryo_4_28_2023\H*')
    all_flds += glob.glob(r'\\192.168.0.7\bbfishmahsa\CGBB_embryo_4_28_2023\H*')
    all_flds = [fld for fld in all_flds if ((get_iH(fld)<=iHM) and (get_iH(fld)>=iHm) and (set_ in os.path.basename(fld)))]
    #all_flds += glob.glob(r'\\192.168.0.6\bbfishjoy4\CGBB_embryo_4_28_2023\P*'+set__)
    
    ### reorder based on hybe
    all_flds = np.array(all_flds)[np.argsort([get_iH(fld)for fld in all_flds])] 
    
    
    
    fovs_fl = save_folder+os.sep+'fovs__'+set_+'.npy'
    if not os.path.exists(fovs_fl):
        all_flds_ = [fld for fld in all_flds if 'lowMER' not in os.path.basename(fld)]
        fls = glob.glob(all_flds_[0]+os.sep+'*.zarr')
        fovs = np.sort([os.path.basename(fl) for fl in fls])
        np.save(fovs_fl,fovs)
    else:
        fovs = np.sort(np.load(fovs_fl))
    fov=None
    if ifov<len(fovs):
        fov = fovs[ifov]
        all_flds = [fld for fld in all_flds if os.path.exists(fld+os.sep+fov)]
    return save_folder,all_flds,fov
def main_f(set_ifov,redo_fits = False,redo_drift=False,redo_decoding=False):
    set_,ifov = set_ifov
    save_folder,all_flds,fov = get_files(set_ifov)
    if fov is not None:
        print("Computing fitting on: "+str(fov))
        print(len(all_flds),all_flds)
        compute_fits(save_folder,fov,all_flds,redo=redo_fits)
        try:
            print("Computing drift on: "+str(fov))
            compute_drift(save_folder,fov,all_flds,set_,redo=redo_drift)
            compute_decoding(save_folder,fov,set_,redo=redo_decoding)
        except:
            print("Failed:",fov,set_)
    return set_ifov

if __name__ == '__main__':
    # start 4 worker processes
    items = [(set_,ifov)for set_ in ['_D7','_D9','_D13','_D14','_D16'][::-1] for ifov in range(1000)]
    items = [(set_,ifov)for set_ in ['_D16'][::-1] for ifov in range(1000)]
                        
    #main_f(['_D16',59])
    #_,flds,_ = get_files(['_D16',59])
    #for fld in flds:
    #    print(fld)
    if True:
        with Pool(processes=4) as pool:
            print('starting pool')
            result = pool.map(main_f, items)

    
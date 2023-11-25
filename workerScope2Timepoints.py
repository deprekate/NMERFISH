
#activate sdeconv&&python C:\Scripts\NMERFISHNEW\workerScope2Timepoints.py

master_analysis_folder = r'C:\Scripts\NMERFISHNEW'
lib_fl = r'C:\Scripts\NMERFISHNEW\codebooks\codebook_0_New_DCBB-300_MERFISH_encoding_2_21_2023.csv'
### Did you compute PSF and median flat field images?
psf_file = r'C:\Scripts\NMERFISHNEW\psfs\psf_cy5_Scope2_final.npy'
flat_field_tag = r'C:\Scripts\NMERFISHNEW\flat_field\Scope2_'
master_data_folders = [r'\\192.168.0.6\bbfish1e3\DCBBL1_03_14_2023_big',
                       r'\\192.168.0.6\bbfish1e4\DCBBL1_03_14_2023_big',
                       r'\\192.168.0.3\bbfishdc9\DCBBL1_03_14_2023_big']
save_folder =r'\\192.168.0.6\bbfish1e3\DCBBL1_03_14_2023_big\MERFISH_AnalysisV2'

iHm=0#1+12*3
iHM=1000

from multiprocessing import Pool, TimeoutError
import time,sys
import os,sys,numpy as np

sys.path.append(master_analysis_folder)
from ioMicro import *

def compute_drift_features(save_folder,fov,all_flds,set_,redo=False,gpu=True):
    fls = [fld+os.sep+fov for fld in all_flds]
    for fl in fls:
        get_dapi_features(fl,save_folder,set_,gpu=gpu,im_med_fl = flat_field_tag+r'med_col_raw3.npz',
                    psf_fl = psf_file)
                    
def get_best_translation_pointsV2(fl,fl_ref,save_folder,set_,resc=5):
    
    obj = get_dapi_features(fl,save_folder,set_)
    obj_ref = get_dapi_features(fl_ref,save_folder,set_)
    tzxyf,tzxy_plus,tzxy_minus,N_plus,N_minus = np.array([0,0,0]),np.array([0,0,0]),np.array([0,0,0]),0,0
    if (len(obj.Xh_plus)>0) and (len(obj.Xh_minus)>0) and  (len(obj_ref.Xh_plus)>0) and (len(obj_ref.Xh_minus)>0):
        X = obj.Xh_plus[:,:3]
        X_ref = obj_ref.Xh_plus[:,:3]
        tzxy_plus,N_plus = get_best_translation_points(X,X_ref,resc=resc,return_counts=True)

        X = obj.Xh_minus[:,:3]
        X_ref = obj_ref.Xh_minus[:,:3]
        tzxy_minus,N_minus = get_best_translation_points(X,X_ref,resc=resc,return_counts=True)
        
        tzxyf = -(tzxy_plus+tzxy_minus)/2
        
    

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
def main_do_compute_fits(save_folder,fld,fov,icol,save_fl,psf,old_method):
    im_ = read_im(fld+os.sep+fov)
    im__ = np.array(im_[icol],dtype=np.float32)
    
    if old_method:
        ### previous method
        im_n = norm_slice(im__,s=30)
        #Xh = get_local_max(im_n,500,im_raw=im__,dic_psf=None,delta=1,delta_fit=3,dbscan=True,
        #      return_centers=False,mins=None,sigmaZ=1,sigmaXY=1.5)
        Xh = get_local_maxfast_tensor(im_n,th_fit=500,im_raw=im__,dic_psf=None,delta=1,delta_fit=3,sigmaZ=1,sigmaXY=1.5,gpu=False)
    else:
        ### new method
        fl_med = flat_field_tag+'med_col_raw'+str(icol)+'.npz'
        if os.path.exists(fl_med):
            im_med = np.array(np.load(fl_med)['im'],dtype=np.float32)
            im_med = cv2.blur(im_med,(20,20))
            im__ = im__/im_med*np.median(im_med)
        else:
            print("Did not find flat field")
        try:
            Xh = get_local_max_tile(im__,th=3600,s_ = 500,pad=100,psf=psf,plt_val=None,snorm=30,gpu=True,
                                    deconv={'method':'wiener','beta':0.0001},
                                    delta=1,delta_fit=3,sigmaZ=1,sigmaXY=1.5)
        except:
            Xh = get_local_max_tile(im__,th=3600,s_ = 500,pad=100,psf=psf,plt_val=None,snorm=30,gpu=False,
                                    deconv={'method':'wiener','beta':0.0001},
                                    delta=1,delta_fit=3,sigmaZ=1,sigmaXY=1.5)
    np.savez_compressed(save_fl,Xh=Xh)
def compute_fits(save_folder,fov,all_flds,redo=False,ncols=4,
                psf_file = psf_file,try_mode=True,old_method=False):
    psf = np.load(psf_file)
    
    for fld in tqdm(all_flds):
        for icol in range(ncols-1):
            tag = os.path.basename(fld)
            save_fl = save_folder+os.sep+fov.split('.')[0]+'--'+tag+'--col'+str(icol)+'__Xhfits.npz'
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
    complete = dec.check_is_complete()
    if complete==0 or redo:
        #compute_drift(save_folder,fov,all_flds,set_,redo=False,gpu=False)
        dec = decoder_simple(save_folder,fov=fov,set_=set_)
        dec.get_XH(fov,set_,ncols=3,nbits=16,th_h=5000)#number of colors match 
        dec.XH = dec.XH[dec.XH[:,-4]>0.25] ### keep the spots that are correlated with the expected PSF for 60X
        dec.load_library(lib_fl,nblanks=-1)
        
        dec.ncols = 3
        dec.get_inters(dinstance_th=2,nmin_bits=4,enforce_color=True,redo=True)
        #dec.get_icodes(nmin_bits=4,method = 'top4',norm_brightness=None,nbits=24)#,is_unique=False)
        get_icodesV2(dec,nmin_bits=4,delta_bits=None,iH=-3,redo=False,norm_brightness=False,nbits=48,is_unique=True)

def get_iH(fld): 
    try:
        return int(os.path.basename(fld).split('_')[0][1:])
    except:
        return np.inf

def get_files(set_ifov,iHm=iHm,iHM=iHM):
    
    if not os.path.exists(save_folder): os.makedirs(save_folder)
    all_flds = []
    for  master_folder in master_data_folders:    
        all_flds += glob.glob(master_folder+r'\H*')
        all_flds += glob.glob(master_folder+r'\P*')
    
    ### reorder based on hybe
    all_flds = np.array(all_flds)[np.argsort([get_iH(fld)for fld in all_flds])] 
    
    set_,ifov = set_ifov
    all_flds = [fld for fld in all_flds if set_ in os.path.basename(fld)]
    
    all_flds = [fld for fld in all_flds if ((get_iH(fld)>=iHm) and (get_iH(fld)<=iHM))]
    
    fovs_fl = save_folder+os.sep+'fovs__'+set_+'.npy'
    if not os.path.exists(fovs_fl):
        folder_map_fovs = all_flds[0]#[fld for fld in all_flds if 'low' not in os.path.basename(fld)][0]
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
        

def compute_main_f(save_folder,all_flds,fov,set_,ifov,redo_fits,redo_drift,redo_decoding,try_mode,old_method):
    print("Computing fitting on: "+str(fov))
    print(len(all_flds),all_flds)
    compute_fits(save_folder,fov,all_flds,redo=redo_fits,try_mode=try_mode,old_method=old_method)
    print("Computing drift on: "+str(fov))
    #compute_drift(save_folder,fov,all_flds,set_,redo=redo_drift)
    compute_drift_features(save_folder,fov,all_flds,set_,redo=False,gpu=True)
    compute_drift_V2(save_folder,fov,all_flds,set_,redo=False,gpu=True)
    compute_decoding(save_folder,fov,set_,redo=redo_decoding)

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
    
    
    
if __name__ == '__main__':
    # start 4 worker processes
    #items = [(set_,ifov)for set_ in ['_set1','_set2','_set3','_set4','_set5']
    #                    for ifov in range(300)]
    dic_f = {'set3': [88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 479, 483, 484, 485, 486, 487, 530, 532, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 573, 574, 575, 576, 577, 578, 579, 580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 598, 599, 600, 601, 602, 603, 604, 605, 606, 607, 608, 653, 654, 655, 657, 660, 680, 681, 682, 683, 684, 685, 686, 687, 688, 689, 690, 691, 692, 693, 694, 695, 696, 697, 698, 699, 700, 701, 702, 704], 'set2': [117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 325, 326, 327, 328, 329, 330, 331, 332, 333, 345, 346, 347, 348, 349, 350, 351, 352, 358, 359, 360, 361, 362, 363, 364, 365, 366, 368, 371, 373, 374, 375, 376, 377, 378, 380, 381, 382, 514, 515, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546, 547, 559, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 573, 574, 575, 576, 577, 578, 579, 580, 581, 582, 583, 584, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 628, 629, 630, 631, 632, 633, 634], 'set6': [133, 184, 185, 186, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 286, 287, 288, 289, 290, 291, 292, 293, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 541, 542, 543, 544, 545, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 567, 568, 569, 570, 571, 572, 573, 580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 595, 596, 597, 598, 599, 600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617], 'set5': [317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 345, 346, 347, 348, 349, 350, 351, 366, 367, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 451, 452, 453, 454, 455, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 502, 503, 629, 630, 655, 666, 668, 669, 670, 671]}
    items = [('_'+set_,ifov) for set_ in dic_f for ifov in dic_f[set_]]
    main_f(items[0],try_mode=False)
    #print("Found decode fls:",len(glob.glob(save_folder+os.sep+'decoded*')))
    if True:
        with Pool(processes=3) as pool:
            print('starting pool')
            result = pool.map(main_f, items)

    #activate sdeconv&&python C:\Scripts\NMERFISHNEW\workerScope2Timepoints.py
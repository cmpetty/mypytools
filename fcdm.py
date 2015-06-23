#!/usr/bin/python
# -*- coding: iso-8859-1 -*-

import sys
import os
import numpy as np
import nibabel as nb
import scipy.stats
import itertools
from scipy import ndimage as nd

'''

Functional connectivity density mapping
Algorithm adapted from Dardo Tomasi, PNAS(2010), vol. 107, no. 21. 9885–9890

'''

def log(s,level=0):
    if level > 0:
        print s

def fcdm(datafile,maskfile,thr):
    data = nb.load(datafile)
    mask = nb.load(maskfile)

    basePath = os.path.dirname(datafile)

    if data.shape[:-1] != mask.shape:
        raise IndexError("Data and Mask are not the same x,y,z shape!")
    #make a masked array and dilate the gm
    mdata = np.ma.array(data.get_data(),mask=np.tile((nd.binary_dilation(mask.get_data()).astype(mask.get_data_dtype()) == 0)[:,:,:,np.newaxis], (1, 1, 1, data.shape[3])))

    #shape holder
    mshape = mdata.shape[:-1]

    #hold results
    fc_dens = np.zeros(mshape, dtype=np.float32)
    
    for x,y,z in itertools.product(xrange(mshape[0]),xrange(mshape[1]),xrange(mshape[2])):
        if mdata[x,y,z].any(): #inmask
            nc = 0
            V = mdata[x,y,z].data

            if np.max(V) > 0:
                nc=1

            nc0=nc

            if nc == 0:
                continue

            c=1
            l3=0

            while c > thr:  #w1
                c=1
                l2=0
                while c > thr: #w2
                    c=1
                    l1=0
                    while c > thr: #w3
                        l1+=1     
                        if (x+l1 < mshape[0]) and (y+l2 < mshape[1]) and (z+l3 < mshape[2]):
                            if not mdata[x+l1,y+l2,z+l3].any(): #not in mask
                                break #to #j1
                        else:
                            break #to j1

                        U=mdata[x+l1,y+l2,z+l3].data
                        c=scipy.stats.pearsonr(U,V)[0]
                    
                        if c > thr:
                            nc+=1
                    #/w3

                    log("got to j1: %d,%d,%d %d,%d,%d nc = %d" % (x,y,z,l1,l2,l3,nc))
                    c=1
                    l1=0

                    while c > thr: #w4
                        l1+=1
                        if (x-l1 >= 0) and (y+l2 < mshape[1]) and (z+l3 < mshape[2]):
                            if not mdata[x-l1,y+l2,z+l3].any(): #not in mask
                                break #to #j2
                        else: 
                            break #to #j2

                        U=mdata[x-l1,y+l2,z+l3].data
                        c=scipy.stats.pearsonr(U,V)[0]

                        if c > thr:
                            nc+=1
                    #/w4

                    log("got to j2: %d,%d,%d %d,%d,%d nc = %d" % (x,y,z,l1,l2,l3,nc))
                    if nc != nc0:
                        l2+=1
                        nc0=nc
                    else:
                        break #to #j9
                #/w2

                log("got to j9: %d,%d,%d %d,%d,%d nc = %d" % (x,y,z,l1,l2,l3,nc))
                c=1
                l2=1

                while c > thr: #w5
                    c=1
                    l1=0
                    while c > thr: #w6
                        l1+=1
                        if (x+l1 < mshape[0]) and (y-l2 >= 0) and (z+l3 < mshape[2]):
                            if not mdata[x+l1,y-l2,z+l3].any(): #not in mask
                                break #to #j3
                        else: 
                            break #to j3

                        U=mdata[x+l1,y-l2,z+l3].data
                        c=scipy.stats.pearsonr(U,V)[0]

                        if c > thr:
                            nc+=1
                    #/w6

                    log("got to j3: %d,%d,%d %d,%d,%d nc = %d" % (x,y,z,l1,l2,l3,nc))
                    c=1
                    l1=0

                    while c > thr: #w7
                        l1+=1
                        if (x-l1 >= 0) and (y-l2 >= 0) and (z+l3 < mshape[2]):
                            if not mdata[x-l1,y-l2,z+l3].any(): #not in mask
                                break #to j4
                        else:
                            break #to j4

                        U=mdata[x-l1,y-l2,z+l3].data
                        c=scipy.stats.pearsonr(U,V)[0]

                        if c > thr:
                            nc+=1
                    #/w7

                    log("got to j4: %d,%d,%d %d,%d,%d nc = %d" % (x,y,z,l1,l2,l3,nc))
                    if nc != nc0:
                        l2+=1
                        nc0=nc
                    else:
                        break #to j10
                #/w5    
            
                log("got to j10: %d,%d,%d %d,%d,%d nc = %d" % (x,y,z,l1,l2,l3,nc))
                if nc != nc0:
                    l3+=1
                    nc0=nc
                else:
                    break #to j11
            #/w1

            log("got to j11: %d,%d,%d %d,%d,%d nc = %d" % (x,y,z,l1,l2,l3,nc))
            c=1
            l3=1

            while c > thr: #w8
                c=1
                l2=0
                while c > thr: #w9
                    c=1
                    l1=0
                    while c > thr: #w10
                        l1+=1
                        if (x+l1 < mshape[0]) and (y+l2 < mshape[1]) and (z-l3 >= 0):
                            if not mdata[x+l1,y+l2,z-l3].any(): #not in mask
                                break #to j5
                        else:
                            break #to j5

                        U=mdata[x+l1,y+l2,z-l3].data
                        c=scipy.stats.pearsonr(U,V)[0]

                        if c > thr:
                            nc+=1
                    #/w10

                    log("got to j5: %d,%d,%d %d,%d,%d nc = %d" % (x,y,z,l1,l2,l3,nc))
                    c=1
                    l1=0

                    while c > thr: #w11
                        l1+=1
                        if (x-l1 >= 0) and (y+l2 < mshape[1]) and (z-l3 >= 0):
                            if not mdata[x-l1,y+l2,z-l3].any(): #not in mask
                                break #to j6
                        else:
                            break #to j6

                        U=mdata[x-l1,y+l2,z-l3].data
                        c=scipy.stats.pearsonr(U,V)[0]

                        if c > thr:
                            nc+=1
                    #/w11

                    log("got to j6: %d,%d,%d %d,%d,%d nc = %d" % (x,y,z,l1,l2,l3,nc))
                    if nc != nc0:
                        l2+=1
                        nc0=nc
                    else:
                        break #to j12
                #/w9

                log("got to j12: %d,%d,%d %d,%d,%d nc = %d" % (x,y,z,l1,l2,l3,nc))
                c=1
                l2=1
                while c > thr: #w12
                    c=1
                    l1=0
                    while c > thr: #w13
                        l1+=1
                        if (x+l1 < mshape[0]) and (y-l2 >= 0) and (z-l3 >= 0):
                            if not mdata[x+l1,y-l2,z-l3].any(): #not in mask
                                break #to j7
                        else:
                            break #to j7

                        U=mdata[x+l1,y-l2,z-l3].data
                        c=scipy.stats.pearsonr(U,V)[0]

                        if c > thr:
                            nc+=1
                    #/w13

                    log("got to j7: %d,%d,%d %d,%d,%d nc = %d" % (x,y,z,l1,l2,l3,nc))
                    c=1
                    l1=0

                    while c > thr: #w14
                        l1+=1
                        if (x-l1 >= 0) and (y-l2 >= 0) and (z-l3 >= 0):
                            if not mdata[x-l1,y-l2,z-l3].any(): #not in mask
                                break #to j8
                        else:
                            break #to j8

                        U=mdata[x-l1,y-l2,z-l3].data
                        c=scipy.stats.pearsonr(U,V)[0]

                        if c > thr:
                            nc+=1
                    #/w14
                
                    log("got to j8: %d,%d,%d %d,%d,%d nc = %d" % (x,y,z,l1,l2,l3,nc))
                    if nc != nc0:
                        l2+=1
                        nc0=nc
                    else:
                        break #to j13

                #/12
                log("got to j13: %d,%d,%d %d,%d,%d nc = %d" % (x,y,z,l1,l2,l3,nc))
                if nc != nc0:
                    l3+=1
                    nc0=nc
                else:
                    break #to j14
            #/w8

            log("got to j14: %d,%d,%d %d,%d,%d nc = %d" % (x,y,z,l1,l2,l3,nc))
            log("%d,%d,%d: r = %f, nc = %d" % (x,y,z,np.float(np.nan_to_num(c)),nc),1)
            fc_dens[x,y,z]=1.*nc
        else:
            #not in mask
            pass
                        
    #save the results
    niftiname = str(os.path.join(basePath,'fcdm.nii.gz'))
    newNii = nb.Nifti1Image(fc_dens,mask.get_affine())
    print "saving %s" % niftiname
    nb.save(newNii,os.path.join(basePath,'fcdm.nii.gz'))
    return niftiname
    

if __name__ == "__main__":

    datafile = None
    maskfile = os.path.join(os.environ['FSLDIR'],'data','standard','MNI152_T1_2mm_brain_pve_1.nii.gz')
    thr = 0.6

    if len(sys.argv) == 1:
        print "Please provide (data, mask, threshold)"
    elif len(sys.argv) > 1:
        datafile = str(sys.argv[1])
        if not os.path.exists(datafile):
            raise Exception("Input file does not exist!")

        if len(sys.argv) > 2:
            maskfile = str(sys.argv[2])
            if not os.path.exists(maskfile):
                raise Exception("Input file does not exist!")

        if len(sys.argv) == 3:
            thr = float(sys.argv[3])

        outname = fcdm(datafile,maskfile,thr)
            


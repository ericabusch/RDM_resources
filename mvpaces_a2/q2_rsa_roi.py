#!/usr/bin/env python
# to be run as ./rsa_roi_attention.py lh FFA
# Options: FFA, LIP, EBA, IPS, PM

import numpy as np
import pylab as pl
from os.path import join as pjoin
import mvpa2.suite as mv
import glob
from scipy.stats import zscore
from mvpa2.mappers.fx import mean_group_sample
from mvpa2.measures import rsa
from scipy.spatial.distance import pdist
import sys

from nilearn import surface

# parse arguments
hemi = sys.argv[1]
region = sys.argv[2]
data_path = '/dartfs-hpc/scratch/psyc164/mvpaces/glm/'
mask_path = '/dartfs-hpc/rc/home/4/f002d44/comp_meth/'
#mask_path = '/dartfs-hpc/scratch/psyc164/mvpaces/masks/'
# ROIs and masks
regions = {
    'FFA': mask_path+ 'fusiform_association-test_z_FDR_0.01.nii.gz',
    'EBA': mask_path+'extrastriate_association-test_z_FDR_0.01.nii.gz',
    'PM': mask_path+'premotor_association-test_z_FDR_0.01.nii.gz',
    'IPS': mask_path+'intraparietal sulcus_association-test_z_FDR_0.01.nii.gz',
    'OP': mask_path+'occipital_parietal_association-test_z_FDR_0.01.nii.gz'
}
# get the desired max activation node for our ROI
mask_file = regions.get(region)
surf = surface.vol_to_surf(img=mask_file, surf_mesh=data_path+'{0}.pial.gii'.format(hemi))
max_node = np.argmax(surf)
n_vertices=40962

subid = [1,12,17,27,32,33,34,36,37,41]
subjs = ['{:0>6}'.format(i) for i in subid]
taxonomy = np.repeat(['bird', 'insect', 'primate', 'reptile', 'ungulate'],4)
behavior = np.tile(['eating', 'fighting', 'running', 'swimming'],5)
conditions = [' '.join((beh, tax)) for beh, tax in zip(behavior, taxonomy)]
radius = 10 
surface = mv.surf.read(pjoin(data_path, '{0}.pial.gii'.format(hemi)))

all_ROI_res = [] 
for sub in subjs:
    # get all our data files for this subj
    ds = None
    prefix = data_path+'sub-rid'+sub
    suffix = hemi+'.coefs.gii'
    fn = prefix + '*' + suffix
    files = sorted(glob.glob(fn))
    for x in range(len(files)):
        if x < 5:
            chunks = [x+1]*20
        else:
            chunks = [x-5+1]*20
        d = mv.gifti_dataset(files[x], chunks=chunks, targets=conditions)
        d.sa['conditions']=conditions
        if ds is None:
            ds = d
        else:      
            ds = mv.vstack((ds,d))
    ds.fa['node_indices'] = range(n_vertices)
    ds.samples = zscore(ds.samples, axis=1)
    mtgs = mean_group_sample(['conditions'])
    mtds = mtgs(ds)

    query = mv.SurfaceQueryEngine(surface, radius, distance_metric='dijkstra', fa_node_key='node_indices')
    query.train(ds)
    dsm = rsa.PDist(square=False)
    print('made dsms')
    sl = mv.Searchlight(dsm, query, roi_ids=query.query_byid(max_node))
    slres = sl(mtds)
    mv.debug.active += ['SLC']
    print('made our sls')
    slres.samples = np.nan_to_num(slres.samples)
    all_ROI_res.append(slres.samples)

all_ROI_res = np.array(all_ROI_res)
# this array is (#participants, 190, #nodes)
#all_ROI_res = np.swapaxes(all_slres, 0, 2)

results = np.mean(all_ROI_res, axis=0)
respath = '/dartfs-hpc/scratch/psyc164/mvpaces/lab2/results/'
resname = 'rsa_eb_roi_{0}_{1}'.format(region, hemi)
np.save(respath+resname, results)

    
    
    
    
    
    
    
    


    
    
    
    
    
    
    
    
    
    
    
    
    
    


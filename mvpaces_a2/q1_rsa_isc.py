#!/usr/bin/env python
# to be run as ./rsa_isc_attention.py lh

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

hemi = sys.argv[1]
subid = [1,12, 17, 27, 32, 33, 34,36,37,41]
subjs = ['{:0>6}'.format(i) for i in subid]
taxonomy = np.repeat(['bird', 'insect', 'primate', 'reptile', 'ungulate'],4)
behavior = np.tile(['eating', 'fighting', 'running', 'swimming'],5)
conditions = [' '.join((beh, tax)) for beh, tax in zip(behavior, taxonomy)]
data_path = '/dartfs-hpc/scratch/psyc164/mvpaces/glm/'
dsm = rsa.PDist(center_data=True)
radius = 9 
surface = mv.surf.read(pjoin(data_path, '{0}.pial.gii'.format(hemi)))
# this is an arbitrary radius and distance metric!
query = mv.SurfaceQueryEngine(surface, radius, distance_metric='dijkstra')
sl = mv.Searchlight(dsm, query)
mv.debug.active += ['SLC']
print('made our sls')

all_slres = [] 
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
    ds.fa['node_indices'] = range(ds.shape[1])
    ds.samples = zscore(ds.samples, axis=1)
    mtgs = mean_group_sample(['conditions'])
    mtds = mtgs(ds)
    slres = sl(mtds)
    slres.samples = np.nan_to_num(slres.samples)
    all_slres.append(slres.samples)

# all_slres has all (190, 40962) RDMs for each subject 
# now we need ISCs
# (12, 190, 40962)
# list of 40962 items (12, 190)
all_slres = np.array(all_slres)
all_slres = np.swapaxes(all_slres, 0, 2)

results = []
for sl_data in all_slres:
    # now i have a 190 by 12 matrix
    sl_data = np.swapaxes(sl_data,0,1)
    # now i have a 12 by 190
    corr_dist = pdist(sl_data, 'correlation')
    corrs = np.subtract(np.ones(corr_dist.shape), corr_dist)
    results.append(np.mean(corrs))
results = np.array(results)    
respath = '/dartfs-hpc/scratch/psyc164/mvpaces/lab2/results/'
resname = 'rsa_sl_isc_'+hemi
np.save(respath+resname, results)

    
    
    
    
    
    
    
    


    
    
    
    
    
    
    
    
    
    
    
    
    
    

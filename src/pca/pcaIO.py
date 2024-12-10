import os

import numpy as np
import pandas as pd
import h5py

from pca.pca import prot_pca, pca_analysis, plot_cumulative_variance, plot_explained_variance

H5PATH = os.path.expanduser('~/cftr2/results/data/pca.hdf5')

class load_pca:
    def __init__(self, grpname=None):
        with h5py.File(H5PATH, 'r') as f:
            if grpname is None:
                pcgrp = f
                while len(pcgrp.keys()) > 0:
                    groups_avail = {group: obj for group, obj in pcgrp.items() if isinstance(obj, h5py.Group)}
                    if len(groups_avail) == 0:
                        break
                    for i, key in enumerate(groups_avail.keys()):
                        print(f'{i}: {key}')
                    print("\n")
                    grpname = list(groups_avail.keys())[int(input('Select from available groups: '))]
                    pcgrp = pcgrp[grpname]
            else:
                pcgrp = f[grpname]

            self.grpname = grpname

            ### TODO: problem arises if the input during calculation does not have all traj_ids and frames included
            self.traj_ids = pcgrp.attrs['traj_ids'][:]
            self.nframes = pcgrp.attrs['nframes'][:]
            self.unit = pcgrp.attrs['unit']

            # Load the data
            eigenvectors = pcgrp['eigenvec']
            self.eigenvec = eigenvectors[:,:]
            # This means eigenvectors are column vectors
            self.eigenvec_dims = [dim.label for dim in eigenvectors.dims]

            self.pcvals = pcgrp['pcvals'][:,:]

            self.variance_ratio = pcgrp['variance_ratio'][:]

            self.xyz_center = pcgrp['xyz_center'][:]

            self.atomselect = pcgrp.attrs['atomselect']
            self.atomstride = pcgrp.attrs['atomstride']

    def pca_df(self, n_pcs=10):
        df = pd.DataFrame()
        df['traj_id'] = np.repeat(self.traj_ids, self.nframes)
        df['timestep'] = np.concatenate([np.arange(n) for n in self.nframes])
        df[['pc'+str(i+1) for i in range(n_pcs)]] = self.pcvals[:,:n_pcs]
        return df
    
    def plot_explained_variance(self, axs, n_pcs=50):
        plot_explained_variance(axs, self.variance_ratio, n_pcs)
        xticks = np.concatenate([[0], np.arange(0, n_pcs+1, 10)[1:]-1])
        axs.set_xticks(xticks)
        axs.set_xticklabels(xticks+1)

    def plot_cumulative_variance(self, axs, n_pcs=50):
        plot_cumulative_variance(axs, self.variance_ratio, n_pcs)
        xticks = np.concatenate([[0], np.arange(0, n_pcs+1, 10)[1:]-1])
        axs.set_xticks(xticks)
        axs.set_xticklabels(xticks+1)

    def predict_transform(self, xyzt: np.ndarray, n_pcs=2):
        # Flatten the atom dimension and xyz dimension
        xyzt = xyzt.reshape(xyzt.shape[0], -1)
        X = (xyzt - self.xyz_center)
        W = self.eigenvec[:,:n_pcs]
        transformed =  np.dot(X, W)
        return transformed

def write_pca(pca: prot_pca, grpname: str, n_pcs=10, unit='A',
              overwrite=False, description=None, **other_data):
    with h5py.File(H5PATH, 'r+') as f:
        if grpname in f.keys():
            if overwrite:
                print(f"Overwriting grpname={grpname} in dataset.")
                # For PCA, it is cleaner to just delete the group and start over
                del f[grpname]
            else:
                print(f"grpname={grpname} already in dataset, ending.")
                return
        
        pcgrp = f.create_group(grpname)
            
        pcgrp.attrs['traj_ids'] = pca.traj_ids
        pcgrp.attrs['nframes'] = np.array(list(pca.nframes.values()))
        pcgrp.attrs['unit'] = unit

        pcgrp.attrs['atomselect'] = pca.atomselect
        pcgrp.attrs['atomstride'] = pca.atomstride
        if description is not None:
            pcgrp.attrs['description'] = description

        # Save eigenvectors
        data = pca.pca.weight_matrix.T
        eigenvectors = pcgrp.create_dataset('eigenvec', data.shape, dtype='float64', maxshape=data.shape)
        eigenvectors[:,:] = data
        # This means eigenvectors are column vectors
        eigenvectors.dims[0].label = 'atom'
        eigenvectors.dims[1].label = 'pc'

        # Save pcvals
        data = pca.pca.pca_output[:,:n_pcs]
        pcvals = pcgrp.create_dataset('pcvals', data.shape, dtype='float64', maxshape=data.shape)
        pcvals[:,:] = data
        pcvals.dims[0].label = 'snapshot'
        pcvals.dims[1].label = 'pc'

        # Save variance ratio
        data = pca.pca.variances
        var = pcgrp.create_dataset('variance_ratio', data.shape, dtype='float32', maxshape=data.shape)
        var[:] = data

        # Save other data such as xyz_center
        for name, data in other_data.items():
            dset = pcgrp.create_dataset(name, data.shape, dtype='float32', maxshape=data.shape)
            dset[:] = data



class load_dpca:
    def __init__(self, grpname=None):
        with h5py.File(H5PATH, 'r') as f:
            if grpname is None:
                pcgrp = f
                while len(pcgrp.keys()) > 0:
                    groups_avail = {group: obj for group, obj in pcgrp.items() if isinstance(obj, h5py.Group)}
                    if len(groups_avail) == 0:
                        break
                    for i, key in enumerate(groups_avail.keys()):
                        print(f'{i}: {key}')
                    grpname = list(groups_avail.keys())[int(input('Select from available groups: '))]
                    pcgrp = pcgrp[grpname]
            else:
                pcgrp = f[grpname]

            self.grpname = grpname

            self.traj_ids = pcgrp.attrs['traj_ids'][:]
            self.nframes = pcgrp.attrs['nframes'][:]

            # Load the data
            eigenvectors = pcgrp['eigenvec']
            self.eigenvec = eigenvectors[:,:]
            # This means eigenvectors are column vectors
            self.eigenvec_dims = [dim.label for dim in eigenvectors.dims]

            self.pcvals = pcgrp['pcvals'][:,:]

            self.variance_ratio = pcgrp['variance_ratio'][:]

    def pca_df(self, n_pcs=10):
        df = pd.DataFrame()
        df['traj_id'] = np.repeat(self.traj_ids, self.nframes)
        df['timestep'] = np.concatenate([np.arange(n) for n in self.nframes])
        df[['pc'+str(i+1) for i in range(n_pcs)]] = self.pcvals[:,:n_pcs]
        return df

def write_dpca(pca: pca_analysis, grpname: str, traj_ids, nframes, resids,
               n_pcs=10, 
               overwrite=False, description=None, **other_data):
    with h5py.File(H5PATH, 'r+') as f:
        if grpname in f.keys():
            if overwrite:
                print(f"Overwriting grpname={grpname} in dataset.")
                # For PCA, it is cleaner to just delete the group and start over
                del f[grpname]
            else:
                print(f"grpname={grpname} already in dataset, ending.")
                return
        
        pcgrp = f.create_group(grpname)
            
        pcgrp.attrs['traj_ids'] = traj_ids
        pcgrp.attrs['nframes'] = nframes
        pcgrp.attrs['resids'] = resids

        if description is not None:
            pcgrp.attrs['description'] = description

        # Save eigenvectors
        data = pca.weight_matrix.T
        eigenvectors = pcgrp.create_dataset('eigenvec', data.shape, dtype='float64', maxshape=data.shape)
        eigenvectors[:,:] = data
        # This means eigenvectors are column vectors
        eigenvectors.dims[0].label = 'rama_param'
        eigenvectors.dims[1].label = 'pc'

        # Save pcvals
        data = pca.pca_output[:,:n_pcs]
        pcvals = pcgrp.create_dataset('pcvals', data.shape, dtype='float64', maxshape=data.shape)
        pcvals[:,:] = data
        pcvals.dims[0].label = 'snapshot'
        pcvals.dims[1].label = 'pc'

        # Save variance ratio
        data = pca.variances
        var = pcgrp.create_dataset('variance_ratio', data.shape, dtype='float32', maxshape=data.shape)
        var[:] = data

        # Save other data such as xyz_center
        for name, data in other_data.items():
            dset = pcgrp.create_dataset(name, data.shape, dtype='float32', maxshape=data.shape)
            dset[:] = data
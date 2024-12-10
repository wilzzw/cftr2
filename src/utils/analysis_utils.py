import os
import json

import numpy as np
import pandas as pd
import mdtraj as md
from netCDF4 import Dataset

from utils.output_namespace import trajectory, grotop4traj
from utils.core_utilities import always_array

# Core class for analysis
# Current version: traj_ids with associated ntsteps
class trajdat:
    def __init__(self, traj_ids, from_nc=True):
        self.traj_ids = always_array(traj_ids)
        self.from_nc = True

def get_ntsteps(traj_id, dat_source, traj=None, nc=None):
    if dat_source == "trajectory":
        if traj is None:
            traj = md.load(trajectory(traj_id), grotop4traj(traj_id))
        nframes = traj.n_frames
    elif dat_source == "netCDF":
        if type(nc) is str:
            # Assume nc given is the path to the ncFile
            dataload = Dataset(nc, "r", format="NETCDF4", persist=True)
            nframes = int(dataload.variables['nframes'][traj_id].data)
            dataload.close()
        else:
            nframes = int(nc.variables['nframes'][traj_id].data)
    return nframes

def make_df(traj_ids, nframes, **label_kw):

    if len(label_kw) > 1:
        raise ValueError('Only one label can be passed')

    # Dataframe initialization
    df = pd.DataFrame()

    if len(label_kw) == 1:
        name, labels = list(label_kw.items())[0]
        # traj_id column
        df['traj_id'] = np.repeat(traj_ids, np.array(nframes)*len(labels))
        # timestep column
        df['timestep'] = np.repeat(np.concatenate([np.arange(n) for n in nframes]), len(labels))
        # helix column
        df[name] = np.tile(labels, np.sum(nframes))

    else:
        # traj_id column
        df['traj_id'] = np.repeat(traj_ids, np.array(nframes))
        # timestep column
        df['timestep'] = np.concatenate([np.arange(n) for n in nframes])
        
    return df   

from scipy.spatial.distance import mahalanobis
from scipy.stats import multivariate_normal

class gaussian2d:
    def __init__(self, mu, Sigma, xedges, yedges):
        self.mu = mu
        self.Sigma = Sigma
        self.xedges = xedges
        self.yedges = yedges

        self.meshgrid = np.meshgrid(self.xedges, self.yedges)
        self.G = multivariate_normal(mean=mu, cov=Sigma)

    def mahdist(self, xy):
        dist = mahalanobis(u=self.mu, v=xy, VI=np.linalg.inv(self.Sigma))
        return dist
    
    def pdf_val_threshold(self, chisq_ddof2=4.61):
        x, y = self.meshgrid
        madist_grid = np.apply_along_axis(self.mahdist, axis=2, arr=np.dstack([x, y]))

        # Threshold
        # P-value of chi-sq at ddof=2

        xwhere, ywhere = np.where(madist_grid <= chisq_ddof2)

        wheremax_madist_sq = np.argmax(madist_grid[xwhere, ywhere])
        xy_max_madist_sq = np.dstack([x, y])[xwhere, ywhere][wheremax_madist_sq]

        return self.G.pdf(xy_max_madist_sq)
    
# Linear regression
from scipy import stats

class linear_regression:
    def __init__(self, x, y, **linregress_kwargs):
        assert len(x) == len(y), 'x and y must have the same length'
        self.x = x
        self.y = y
        self.results = stats.linregress(self.x, self.y, **linregress_kwargs)

        # My add-ons
        self.n = len(x)
        self.y_pred = self.results.slope*x + self.results.intercept
        self.residuals = y - self.y_pred
        self.df = len(x)-2
        self.mse = np.sum(self.residuals**2) / self.df

    def predict(self, x, alpha=0.05):
        # https://online.stat.psu.edu/stat501/lesson/3/3.3
        t_stat = stats.t.ppf(1-(alpha/2), self.df) # Two tailed
        y_pred = self.results.slope*x + self.results.intercept
        y_pred_stderr = t_stat * np.sqrt(self.mse * (1 + 1/self.n + (x-np.mean(self.x))**2 / np.sum((self.x-np.mean(self.x))**2)))
        return y_pred, y_pred_stderr
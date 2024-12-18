import os

import numpy as np
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.analysis import hole2

from database.query import traj2grotop
from utils.output_namespace import trajectory, grotop4traj, hole2_data


def run_hole2(u: mda.Universe, rmtmp=True, cpoint=[23.55, 56.32, 106.37], **run_kwargs):
    # TODO: HOLE2 parameters should be more visible
    ha = hole2.HoleAnalysis(u, select='protein and resid 1:380 846:1173 and prop z>90', 
                            cpoint=cpoint, cvect=[0,0,1], ignore_residues=['UNX'],
                            executable='~/hole2/exe/hole')
    ha.run(verbose=True, **run_kwargs)
    ha.create_vmd_surface(filename=f'hole.vmd')
    if rmtmp:
        ha.delete_temporary_files()
    return ha

def run_hole2_trajid(traj_id, save=False, rmtmp=True, **run_kwargs):
    # Load the trajectory as a Universe
    u = mda.Universe(grotop4traj(traj_id), trajectory(traj_id))
    ha = run_hole2(u, rmtmp=rmtmp, **run_kwargs)
    if save:
        np.save(hole2_data(traj_id), ha.results.profiles)
    return ha

def load_holedata(traj_id):
    return np.atleast_1d(np.load(hole2_data(traj_id), allow_pickle=True))[0]

def make_hole_df(traj_ids):
    hole2_df = []
    for t in traj_ids:
        traj_data = load_holedata(t)
        traj_hole2_df = []
        for f, frame_data in traj_data.items():
            # Filter out dummies below z=90 A
            filter_where,  = np.where(frame_data['rxn_coord'] >= 90)
            traj_hole2_df.append(frame_data['radius'][filter_where][::10])
        traj_hole2_df = pd.DataFrame(traj_hole2_df)
        hole2_df.append(traj_hole2_df)

    hole2_df = pd.concat(hole2_df, ignore_index=True)
    # Rename the columns to +90
    hole2_df.columns = hole2_df.columns.astype(int) + 90

    hole2_df = hole2_df.loc[:,:150]
    return hole2_df
import os
from sys import argv

import numpy as np
import pandas as pd
import MDAnalysis as mda

from channel.translocate import transloc_iontraj
from utils.output_namespace import trajectory, grotop4traj
from database.query import traj_group

if len(argv) == 2:
    # Allow for the script to be restarted from a certain trajectory
    restart = int(argv[1])
elif len(argv) > 2:
    raise ValueError("Too many arguments provided")

traj_ids = np.array(traj_group(3))
output_pkl = os.path.expanduser(f"results/data/transloc_ionenv.pkl")
trajectory_loc = os.path.expanduser("~/project/research/trajectories/")



main = transloc_iontraj(traj_ids)
transloc_df = main.timeseries_transloc_df(xcenter=25, ycenter=50, radius=(300**0.5), zmin=90, zmax=170)

dt = 1000
transloc_df['start'] = transloc_df['start'] * transloc_df['timestep'] // dt
transloc_df['end'] = transloc_df['end'] * transloc_df['timestep'] // dt
transloc_df['timestep'] = dt


try:
    restart
except NameError:
    transloc_ionenv = pd.DataFrame(columns=['traj_id', 'timestep', 'transloc_id', 'atom_index', 'coord_a', 'coord_r'])
else:
    transloc_ionenv = pd.read_pickle(output_pkl)


for traj_id, df in transloc_df.groupby('traj_id'):
    print(f"Processing trajectory {traj_id}...")
    u = mda.Universe(grotop4traj(traj_id), trajectory(traj_id, path2traj=trajectory_loc, align_program=""))

    atom_index = df['atom_index'].values
    # transloc_id
    transloc_id = df['transloc_id'].values
    # Define time interval to be analyzed for the ion
    start, end = df[['start', 'end']].values.T

    try:
        restart
    except NameError:
        pass
    else:
        if traj_id <= restart:
            continue

    for a, id, start, end in zip(atom_index, transloc_id, start, end):
        ion_environ = u.select_atoms(f"around 3 index {a}", updating=True)
        ion = u.select_atoms(f"index {a}")

        # TODO: still need to potentially account for the different timestep values
        for timestep in range(start, end+1):
            snapshot = u.trajectory[timestep]
            coord_a = sorted(ion_environ.atoms.indices)
            residues_surround = ion_environ.residues
            coord_r = ["".join([r.resname, str(r.resid)]) for r in residues_surround]
            residues_surround.resids
            # For test/troubleshooting purposes
            ionxyz = ion.positions.flatten()
            transloc_ionenv = pd.concat([transloc_ionenv, pd.DataFrame([[traj_id, timestep, id, a, coord_a, coord_r]], 
                                         columns=transloc_ionenv.columns)], ignore_index=True)
    
    transloc_ionenv.to_pickle(output_pkl)
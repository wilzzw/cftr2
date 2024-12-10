import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from plot.plot_utilities import hist2d

def vmdtcl(func):
    def write(script_name, *args, **kwargs):
        with open(os.path.expanduser(f"~/cftr2/workspace/{script_name}"), 'w') as script:
            tf = func(*args, **kwargs)
            script.write("mol new data/topologies/1_init.gro type " + '{gro}' + " first 0 last -1 step 1 waitfor -1\n")
            for t, f in tf[['traj_id', 'timestep']].values:    
                script.write(f"mol new trajectories/{t}_md_fixed_dt1000_mda.xtc type " +'{xtc}' + f" -molid top first {f} last {f} step 1 waitfor -1 top\n")
            script.write("animate delete beg 0 end 0 skip 0 top\n")
        return tf
    return write

# Sample some snapshots
@vmdtcl
def sample_snapshots(metrics_df: pd.DataFrame, n_sample: int, random_state: int = 100, **metric_ranges):
    # Square range; it is not necessary to have a pretty geometry
    sampled_snapshots = metrics_df.query(' & '.join([f'{k} >= {v[0]} & {k} <= {v[1]}' for k, v in metric_ranges.items()]))
    sampled_snapshots = sampled_snapshots.sample(n_sample, random_state=random_state)
    sampled_snapshots = sampled_snapshots.sort_values(by=['traj_id', 'timestep'])[['traj_id', 'timestep']]
    
    return sampled_snapshots
    

# Pick snapshots within a state peak based on contour
def pick_state_from_contour(hist: hist2d, contour_cutoff, state_select, df, 
               n_sample=100, random_state=100, visualize=False):
    gridpts_inrange = (hist.dens > contour_cutoff)
    pts_inrange = gridpts_inrange[hist.xbin_index, hist.ybin_index]
    snapshot_select = np.all([pts_inrange, state_select], axis=0)

    sampled_snapshots = df.loc[snapshot_select].sample(n_sample, random_state=random_state)
    if visualize:
        fig, axs = plt.subplots()
        hist.dens2d_preset(axs, lw=0.5)
        axs.scatter(hist.x[snapshot_select], hist.y[snapshot_select])

    sampled_snapshots = sampled_snapshots.sort_values(by=['traj_id', 'timestep'])[['traj_id', 'timestep']]

    return sampled_snapshots
#This place stores the names of the output files so to avoid remembering how I named the output files (keep track of output file naming invariant of the information generated.. you know what I mean?)
#Please remember to pass specific arguments to the functions after importing.
#import numpy as np
import os

from database.query import traj2grotop, gro2grotop, traj2tprtop, get_trajattr
from utils.core_utilities import nmod

def trajectory(traj_id, path2traj="~/cftr2/trajectories/", align_program="mda", dt=None, suffix=""):
    core_trajname = f"{traj_id}_md_fixed"

    if dt is None:
        dt = get_trajattr(traj_id, "dt")

    trajfile = f"{core_trajname}{nmod(dt, insertion='dt')}{nmod(align_program)}{nmod(suffix, connect='-')}.xtc"
    return os.path.expanduser(path2traj + trajfile)

def init_gro(gro_id, literal_homedir=True):
    groloc = f"~/cftr2/data/topologies/{gro_id}_init.gro"
    if literal_homedir:
        groloc = os.path.expanduser(groloc)
    if gro_id != gro2grotop(gro_id):
        print("Warning: the use of xtc as init.gro will be deprecated soon!")
        groloc += ".xtc"
    return groloc

def tprtop(tpr_id):
    return os.path.expanduser(f"~/cftr2/data/topologies/{tpr_id}_dummy.tpr")

def grotop4traj(traj_id):
    return init_gro(gro_id=traj2grotop(traj_id)) 

def tprtop4traj(traj_id):
    print("Be very very careful when using MDAnalysis! When topology is tpr, the residue numbers are renumbered to start from 1.")
    print("When topology is gro, the residue numbers are the same as in the original gro file.")
    return tprtop(traj2tprtop(traj_id))

def aligned_pdbfile(pdb_code: str):
    return os.path.expanduser(f"~/cftr2/data/topologies/{pdb_code.lower()}_aligned.pdb")

def hole2_data(traj_id):
    return os.path.expanduser(f'~/cftr/results/data/{traj_id}_hole2.npy')
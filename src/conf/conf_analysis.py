import json
import os

import numpy as np
import pandas as pd
import mdtraj as md
from netCDF4 import Dataset

from rms.align_utils import align2parts
from database.query import traj2grotop
from utils.output_namespace import init_gro
from utils.core_utilities import always_array


central_atoms_file = open(os.path.expanduser('~/cftr2/data/sidechainCentralAtoms.json'))
central_atoms = json.loads(central_atoms_file.read())
central_atoms_file.close()

def dist(xyz1, xyz2):
    distance = np.linalg.norm(xyz1 - xyz2, axis=-1)
    return distance

class xyzt:
    def __init__(self, traj_ids, from_protein_model_id=1, close_after_init=True):
        """ 
        Parameters:
            traj_ids: self-explanatory; can be array type
            from_protein_model_id: protein_model_id of the protein model on which the resIDs are based
            close_after_init: Whether to close the NC file or not (continue retrieving information or not)
        """
        self.traj_ids = always_array(traj_ids)
        self.from_protein_model_id = from_protein_model_id

        # Read nc
        self.open_nc()

        # Modifies traj_ids, because not all of them have data in the nc
        traj_ids_avail = self.prot_xyz.variables['traj_id'][:]
        traj_ids_avail = traj_ids_avail[~traj_ids_avail.mask]
        exclude_traj_ids = np.setdiff1d(self.traj_ids, traj_ids_avail)
        # if len(exclude_traj_ids) > 0:
        #     print("Excluding traj_ids that have no records in NC: {}".format(exclude_traj_ids))

        self.traj_ids_input = self.traj_ids
        self.traj_ids = np.intersect1d(self.traj_ids, traj_ids_avail)
        self.traj_ids_excluded = exclude_traj_ids

        # New attribute: number of frames registered in NC
        # NC default should be zero
        self.nframes = {t: int(self.prot_xyz.variables['nframes'][t].data) for t in self.traj_ids}

        # gro_ids of topology gro file for all trajectories (no repeat)
        self.grotop_ids = sorted(set([traj2grotop(t) for t in self.traj_ids]))

        # Load ref topologies: mdtraj trajectory object
        self.load_topdf()
        # self.load_refs()
        
        # # Get sidechain central atom indices
        # self.sc_central_index()

        if close_after_init:
            self.close_nc()

    def load_refs(self):
        # Load using mdtraj of the grotop gro files, used for topology
        self.refs = {grotop_id: md.load(init_gro(grotop_id)) for grotop_id in self.grotop_ids}

    # Generally if load_topdf(), then load_refs() is not needed
    def load_topdf(self):
        # Load some useful topology info into dataframe
        self.topdf = {}
        for grotop_id in self.grotop_ids:
            topdf, _ = md.load(init_gro(grotop_id)).top.to_dataframe()
            topdf['grotop_id'] = grotop_id
            self.topdf[grotop_id] = topdf

    # Get xyz for given aindex
    # Optionally assign resids
    # Provide align_aindex and align_xyz to align the coordinates
    def getcoords(self, traj_id, aindex, df=True, 
                  align_aindex=None, align_xyz=None):
        # aindex can be array
        aindex = always_array(aindex)

        # Actual aindex to retrieve
        # TODO: could be more memory efficient by not repicking the same aindex
        if align_aindex is not None:
            align_aindex = always_array(align_aindex)
            retrieve_aindex = np.concatenate([aindex, align_aindex])
        else:
            retrieve_aindex = aindex

        nframes = self.nframes.get(traj_id)
        # TODO: Might be better implemented better to handle units?
        coordinates = self.prot_xyz.variables['coordinate'][traj_id, :nframes, retrieve_aindex, :] * 10 

        # Handle structural alignment
        if align_aindex is not None:
            if align_xyz is None:
                raise ValueError("align_xyz must be provided if align_aindex is provided")
            
            assert align_xyz.shape == (len(align_aindex), 3)
            # Pad some dummy coordinates
            align_xyz = np.pad(align_xyz, ((len(aindex),0),(0,0)), constant_values=0)

            # Align the coordinates
            coordinates = align2parts(coordinates, align_xyz, 
                                      np.arange(len(aindex), len(aindex)+len(align_aindex)), 
                                      np.arange(len(aindex), len(aindex)+len(align_aindex)))
            # Trim off the alig_aindex part
            coordinates = coordinates[:, :len(aindex), :]

        if not df:
            return coordinates

        # Initialize an empty dataframe for this trajectory's xyz's
        traj_xyzdf = pd.DataFrame()

        # Last dimension should have a length=3 (x, y, z); 3 is less wordy
        coordinates = coordinates.reshape(nframes*len(aindex), 3)

        # Insert xyz data into the df
        traj_xyzdf['traj_id'] = np.repeat(traj_id, nframes*len(aindex))
        traj_xyzdf['timestep'] = np.repeat(np.arange(nframes), len(aindex))
        traj_xyzdf['aindex'] = np.tile(aindex, nframes)

        # Get resids for the selected aindex
        # TODO: This line is very slow
        resid = self.topdf[traj2grotop(traj_id)].iloc[aindex]['resSeq'].values
        traj_xyzdf['resid'] = np.tile(resid, nframes)

        traj_xyzdf[['x','y','z']] = coordinates

        return traj_xyzdf

    def open_nc(self):
        self.prot_xyz = Dataset(os.path.expanduser("~/cftr2/data/xyz/protein_xyz.nc"), "r", format="NETCDF4", persist=True)

    def close_nc(self):
        self.prot_xyz.close()



class protca(xyzt):
    def __init__(self, traj_ids, resids, **conf_kwargs):
        super().__init__(traj_ids=traj_ids, **conf_kwargs)

        self.resids = always_array(resids)

    def load_cainfo(self, **align_kwargs):

        self.open_nc()
        # CA info
        self.proc_ca()
        # self.caindex()
        
        # Collect alpha carbon xyz coordinates
        self.get_cacoords(**align_kwargs)

        # Close gracefully?
        self.close_nc()

    def proc_ca(self):
        # Actual available resids for each grotop
        # Dict comp won't work because of the query; use for loop
        self.resids_avail = {}
        self.rcaindex = {}
        for grotop_id, df in self.topdf.items():
            self.resids_avail[grotop_id] = df.query('resSeq in @self.resids')['resSeq'].unique()
            # Get residue alpha carbon atom indices
            self.rcaindex[grotop_id] = df.query('resSeq in @self.resids & name == "CA"').index.values

    # Get alpha carbons' coordinates
    def get_cacoords(self, **align_kwargs):
        # Collect alpha carbon xyz coordinates
        self.ca_coord_set = []

        if len(align_kwargs) > 0:
            align_select = align_kwargs.get('align_select')
            align_xyz = align_kwargs.get('align_xyz')

            # We need the refs to parse the selection to atom indices
            self.load_refs()

            align_aindex = {grotop_id: self.refs[grotop_id].top.select(align_select) for grotop_id in self.refs.keys()}
        else:
            align_aindex = {}
            align_xyz = None

        for t in self.traj_ids:
            rcaindex = self.rcaindex[traj2grotop(t)]

            # Align if necessary
            traj_xyzdf = self.getcoords(t, rcaindex, align_aindex=align_aindex.get(traj2grotop(t)), align_xyz=align_xyz)
            self.ca_coord_set.append(traj_xyzdf)

        # Concat into a larger dataframe
        self.ca_coord_set = pd.concat(self.ca_coord_set, ignore_index=True)

    def compute_dist(self, resid1, resid2):
        # traj (t) and frame (f)
        tf = self.ca_coord_set.query('resid == @resid1')[['traj_id','timestep']]

        r1xyz = self.ca_coord_set.query('resid == @resid1')[['x','y','z']].values
        r2xyz = self.ca_coord_set.query('resid == @resid2')[['x','y','z']].values

        distance = dist(r1xyz, r2xyz)

        dist_df = tf
        dist_df['dist'] = distance

        return dist_df

    # Calc distance between a pair of residues reported by their alpha carbons
    def cadist_rpair(self, resid_pair):
        # Make sure r1 is the smaller one
        r1, r2 = np.sort(resid_pair)

        # Get xyz for the two ca atoms from self.ca_coord_set
        r1xyz = self.ca_coord_set.query('resid == @r1')[['x','y','z']]
        r2xyz = self.ca_coord_set.query('resid == @r2')[['x','y','z']]
        assert len(r1xyz) == len(r2xyz)

        # Prepare distance dataframe
        dist_df = pd.DataFrame()
        # Preserve traj_id information
        dist_df['traj_id'] = self.ca_coord_set.iloc[r1xyz.index]['traj_id']
        # Preserve timestep info for ease of query
        dist_df['timestep'] = self.ca_coord_set.iloc[r1xyz.index]['timestep']
        # Register the pair's resid numbers
        dist_df['r1'] = np.repeat(r1, len(r1xyz))
        dist_df['r2'] = np.repeat(r2, len(r2xyz))
        # Register distances
        dist_df['dist'] = dist(r1xyz.to_numpy(), r2xyz.to_numpy())

        # Reset index for dist_df
        dist_df.reset_index(drop=True, inplace=True)

        return dist_df


class sc_central(xyzt):
    def __init__(self, traj_ids, resids, **conf_kwargs):
        super().__init__(traj_ids=traj_ids, **conf_kwargs)

        self.resids = always_array(resids)

    def load_sccinfo(self, **align_kwargs):

        self.open_nc()
        self.proc_sc()

        self.get_sccoords(**align_kwargs)

        # Close gracefully?
        self.close_nc()

    def proc_sc(self):
        # Actual available resids for each grotop
        self.resids_avail = {}
        self.rscindex = {}
        for grotop_id, df in self.topdf.items():
            resids_avail = df.query('resSeq in @self.resids')['resSeq'].unique()
            self.resids_avail[grotop_id] = resids_avail
            
            resnames = df.query('resSeq in @self.resids & name == "CA"')['resName']
            central_atoms_avail = [central_atoms[resname] for resname in resnames]

            rscindex = []
            for a, r in zip(central_atoms_avail, resids_avail):
                rscindex.append(df.query('resSeq == @r & name == @a').index[0])
            self.rscindex[grotop_id] = np.array(rscindex).flatten()


    # Get sidechain central atom' coordinates
    def get_sccoords(self, **align_kwargs):
        # Collect sidechain central atom xyz coordinates
        self.sc_coord_set = []

        if len(align_kwargs) > 0:
            align_select = align_kwargs.get('align_select')
            align_xyz = align_kwargs.get('align_xyz')

            # We need the refs to parse the selection to atom indices
            self.load_refs()

            align_aindex = {grotop_id: self.refs[grotop_id].top.select(align_select) for grotop_id in self.refs.keys()}
        else:
            align_aindex = {}
            align_xyz = None

        for t in self.traj_ids:
            rscindex = self.rscindex[traj2grotop(t)]

            # Align if necessary
            traj_xyzdf = self.getcoords(t, rscindex, align_aindex=align_aindex.get(traj2grotop(t)), align_xyz=align_xyz)
            self.sc_coord_set.append(traj_xyzdf)
        
        # Concat into a larger dataframe
        self.sc_coord_set = pd.concat(self.sc_coord_set, ignore_index=True)    

    def compute_dist(self, resid1, resid2):
        # traj (t) and frame (f)
        tf = self.sc_coord_set.query('resid == @resid1')[['traj_id','timestep']]

        r1xyz = self.sc_coord_set.query('resid == @resid1')[['x','y','z']].values
        r2xyz = self.sc_coord_set.query('resid == @resid2')[['x','y','z']].values

        distance = dist(r1xyz, r2xyz)

        dist_df = tf
        dist_df['dist'] = distance

        return dist_df
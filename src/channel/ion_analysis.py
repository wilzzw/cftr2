import os

import numpy as np
import pandas as pd

from channel.translocate import ion_xyz, incylinder

def ion_dens3d(xyz, bin_width, range_xyz, density=True):
    bins_xyz = list(map(lambda x: int((x[1]-x[0])*bin_width), range_xyz))
    dens3d, dim_edges = np.histogramdd(xyz, bins=bins_xyz, range=range_xyz, density=True)
    return dens3d, dim_edges

class ion_occupancy(ion_xyz):
    def __init__(self, traj_ids, ion='CLA', readstride=50, 
                 cylinder_kwargs={'xcenter':25, 'ycenter':50, 'radius':300**0.5, 'zmin':110, 'zmax':150}):
        super().__init__(traj_ids, ion, readstride)

        self.xcenter = cylinder_kwargs['xcenter']
        self.ycenter = cylinder_kwargs['ycenter']
        self.radius = cylinder_kwargs['radius']
        self.zmin = cylinder_kwargs['zmin']
        self.zmax = cylinder_kwargs['zmax']
        
        # Get coordinates of CL ions
        self.get_allcoords()
        self.ion_coord_set["incylinder"] = incylinder(self.ion_coord_set[['x','y','z']].values, 
                                                      **cylinder_kwargs)[0]
        
    def label_ion_coord_set(self, label_df: pd.DataFrame):
        self.ion_coord_set = pd.merge(self.ion_coord_set, label_df, on=["traj_id", "timestep"])

    def calc_occupancy(self, **label_kwarg):
        occupancy_collect = []

        query_strings = []
        for label, value in label_kwarg.items():
            query_strings.append(f"{label} == {value}")
        
        ion_coord_subset = self.ion_coord_set.query(' & '.join(query_strings))

        for tf, df in ion_coord_subset.groupby(["traj_id", "timestep"]):
            timestep, frame = tf
            occupancy_collect.append((timestep, frame, len(df.query("incylinder"))))

        self.occupancy = pd.DataFrame(occupancy_collect, columns=["traj_id", "timestep", "nions"])

# Ion coordination analysis
protein_resnames = ["ASP", "GLU", "LYS", "ARG", "HIS", "HSD", "TYR", "TRP", "PHE", "ASN", "GLN", "SER", "THR", "MET", "ALA", "VAL", "LEU", "ILE", "PRO", "GLY", "CYS"]
other_resnames = ["POPC", "TIP3", "SOD", "CLA"]

# MDTraj convention
polar_hydrogens = {"R": ["HE", "HH11", "HH12", "HH21", "HH22"],
                   "K": ["HZ1", "HZ2", "HZ3"],
                   "H": ["HD1", "HE2"],
                   "S": ["HG"],
                   "T": ["HG1"],
                   "N": ["HD21", "HD22"],
                   "Q": ["HE21", "HE22"],
                   "Y": ["HH"],
                   "W": ["HE1"]}

# Helper functions

# MdTraj convention
# Takes a list of atom indices in reference to the topology and returns the residue names of the residues that contain the atoms
# Residues without sidechain atoms are excluded
def _with_protein_sidechain(atom_indices, topology):
    residue_sidechain_involved = [topology.atom(a).residue for a in atom_indices if topology.atom(a).is_sidechain]
    resname_sidechain_involved = list(set(["".join([r.name, str(r.resSeq)]) for r in residue_sidechain_involved]))
    return resname_sidechain_involved

def _parse_atoms(aindex, topology):
    return [topology.atom(a) for a in aindex]

def _polar_atoms_in_sidechain(atoms):
    sidechains_with_polar = [atom for atom in atoms if atom.residue.code in polar_hydrogens]
    return [atom for atom in sidechains_with_polar if atom.name in polar_hydrogens[atom.residue.code]]

# Helper function to count the number of residues of a certain resname in a list
def count_residues(residue_list, resname):
    return len([res for res in residue_list if res.startswith(resname)])

class ion_coordination:
    def __init__(self):
        # Load the ion coordination data
        self.ion_coordination_df = pd.read_pickle(os.path.expanduser("~/cftr2/results/data/transloc_ionenv.pkl"))

        # Change datatypes
        self.ion_coordination_df['traj_id'] = self.ion_coordination_df['traj_id'].astype(int)
        self.ion_coordination_df['timestep'] = self.ion_coordination_df['timestep'].astype(int)
        self.ion_coordination_df['transloc_id'] = self.ion_coordination_df['transloc_id'].astype(int)
        self.ion_coordination_df['atom_index'] = self.ion_coordination_df['atom_index'].astype(int)
        # Unasigned values: z
        self.ion_coordination_df['z'] = -1e5

        self.ion_coordination_df['coord_r'] = self.ion_coordination_df['coord_r'].apply(np.unique)

        # transloc_ids and traj_ids analyzed in the dataset
        transloc_ids_analyzed = self.ion_coordination_df['transloc_id'].unique()
        traj_ids_analyzed = self.ion_coordination_df['traj_id'].unique()

        # Get the xyz coordinates of the ions
        self.ion_xyz = ion_xyz(traj_ids=traj_ids_analyzed, ion='CLA', readstride=50)

        # Adding z-coordinate of permeant ions to the dataframe
        t = 0
        for i in transloc_ids_analyzed:
            # print(f"Adding z-coordinate for transloc_id: {i}")
            subdf = self.ion_coordination_df.query("transloc_id == @i")
            traj_id = subdf['traj_id'].unique()[0]
            aindex = subdf['atom_index'].unique()[0]
            timesteps = subdf['timestep'].values

            # Get the coordinates of the ion
            # If the trajectory is different, call the function to get the coordinates
            # Otherwise, don't need to call it again to repeat reading the coordinate data file
            if traj_id != t:
                xyz, ion_aindex = self.ion_xyz._get_coords(traj_id)
                t = traj_id
            
            zcoords_transloc = xyz[timesteps, ion_aindex == aindex, 2]
            self.ion_coordination_df.loc[subdf.index, 'z'] = zcoords_transloc

        parsed_atoms = self.ion_coordination_df.apply(lambda row: _parse_atoms(row['coord_a'], self.ion_xyz.refs[1].top), axis=1)
        self.ion_coordination_df["all_atoms"] = parsed_atoms

        sidechain_atoms = parsed_atoms.apply(lambda row: [atom for atom in row if atom.is_sidechain])
        self.ion_coordination_df["sidechain_atoms"] = sidechain_atoms

        polar_sidechain_atoms = sidechain_atoms.apply(_polar_atoms_in_sidechain)
        self.ion_coordination_df["polar_sidechain_atoms"] = polar_sidechain_atoms

        # H atoms are not considered backbone atoms by MDTraj; we have to fix this
        backbone_atoms = parsed_atoms.apply(lambda row: [atom for atom in row if atom.is_backbone or atom.name == 'H'])
        self.ion_coordination_df["backbone_atoms"] = backbone_atoms

        other_atoms = self.ion_coordination_df.apply(lambda row: [atom for atom in row["all_atoms"] if atom not in row["polar_sidechain_atoms"]+row["backbone_atoms"]], axis=1)
        self.ion_coordination_df["other_atoms"] = other_atoms

        # # protein_residue_combo = ["|".join(residue_list) for residue_list in self.ion_coordination_df['coord_r'].apply(lambda x: _protein_r(x))]
        # # self.ion_coordination_df["protein_r_combo"] = protein_residue_combo

        all_protein_residues = [atom.residue for atom in np.concatenate(self.ion_coordination_df['all_atoms']) if atom.residue.is_protein]
        self.all_protein_residues = sorted(set([residue.code + str(residue.resSeq) for residue in all_protein_residues]))

        # other_residues = [atom.residue for atom in np.concatenate(self.ion_coordination_df['all_atoms']) if not atom.residue.is_protein]
        # other_residues = sorted(set([residue.name for residue in other_residues]))
        # Create rlist; ignores protein_resnames that are not found in all_protein_residues_found
        self.rlist = np.concatenate([self.all_protein_residues, other_resnames])

        # Initialize the dataframe
        for resname in self.rlist:
            self.ion_coordination_df[resname] = 0

    def calc_polar(self, sc_only=True):
        self.polar_df = self.ion_coordination_df[['traj_id', 'timestep', 'transloc_id', 'atom_index', 'z']]
        # Calculate other entities first, like water
        for resname in other_resnames:
            self.polar_df[resname] = self.ion_coordination_df['coord_r'].apply(lambda x: count_residues(x, resname))

        # Now, deal with protein residues
        for resname in self.all_protein_residues:
            sidechain_involved = self.ion_coordination_df["polar_sidechain_atoms"].apply(lambda x: resname in [atom.residue.code + str(atom.residue.resSeq) for atom in x])
            if sc_only:
                self.polar_df[resname] = sidechain_involved.astype(int)
            else:
                backbone_involved = self.ion_coordination_df["backbone_atoms"].apply(lambda x: resname in [atom.residue.code + str(atom.residue.resSeq) for atom in x])
                self.polar_df[resname] = (sidechain_involved | backbone_involved).astype(int)

    def calc_nonpolar(self, exclude_polar=True):
        self.nonpolar_df = self.ion_coordination_df[['traj_id', 'timestep', 'transloc_id', 'atom_index', 'z']]

        # Only consider protein residues
        other_protein_atoms = self.ion_coordination_df["other_atoms"].apply(lambda x: [atom for atom in x if atom.residue.is_protein])
        for resname in self.all_protein_residues:
            sidechain_involved = other_protein_atoms.apply(lambda x: resname in [atom.residue.code + str(atom.residue.resSeq) for atom in x])
            
            if exclude_polar:
                self.nonpolar_df[resname] = (sidechain_involved & ~self.polar_df[resname]).astype(bool)
            else:
                self.nonpolar_df[resname] = sidechain_involved.astype(int)

    def prepare(self, zmin=90, zmax=150, bin_width=1):
        self.zmin = zmin
        self.zmax = zmax
        self.bin_width = bin_width

        # Put the entries into bins based on z-coordinate
        self.bin_assign = np.digitize(self.ion_coordination_df['z'], bins=np.arange(zmin,zmax+bin_width,bin_width))
        self.polar_df['bin_index'] = self.bin_assign
        self.nonpolar_df['bin_index'] = self.bin_assign

        self.bin_index = np.sort(np.unique(self.bin_assign))[1:-1]
        self.bin_z = np.arange(zmin,zmax,bin_width) + bin_width/2

    def _analyze(self, df, query: str, **query_kwargs):
        # Create variables from the query_kwargs
        for var, value in query_kwargs.items():
            exec(f"{var} = {value}")
        subdf = df.query(query)

        resid_bind = {r: [] for r in self.rlist}

        for i in self.bin_index:
            # For testing
            # print(ion_coord_path_subdf.query("bin_index == @i")[rlist].mean().sort_values(ascending=False)[:5])
            for r in self.rlist:
                try:
                    resid_in_bin = subdf.query("bin_index == @i")[r]
                except KeyError:
                    resid_bind[r].append(None)
                else:
                    avg_nbind = subdf.query("bin_index == @i")[r].mean()
                    resid_bind[r].append(avg_nbind)

        return resid_bind
    
    # A bit awkward
    def analyze(self, query: str="traj_id > 0", **query_kwargs):
        self.resid_bind = self._analyze(self.polar_df, query, **query_kwargs)
        self.resid_bind_nonpolar = self._analyze(self.nonpolar_df, query, **query_kwargs)
    
    def plot_resid_coord_contribute(self, axs, resname, title=True, **plot_kwargs):
        if title:
            axs.set_title(resname)

        axs.plot(self.bin_z, self.resid_bind[resname], **plot_kwargs)
        axs.set_ylim(0,1)
        axs.set_xlim(self.zmin,self.zmax)
        axs.grid(True, ls='--')
        axs.set_xticks(np.arange(self.zmin,self.zmax+5,5))
        axs.set_xticklabels(np.arange(self.zmin,self.zmax+5,5)-130)

        axs.set_xlabel(r"z [$\mathrm{\AA}$]", fontsize=16)
        axs.set_ylabel(r"$n_{solv}$(Cl-)", fontsize=16)


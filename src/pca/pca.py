import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA as IPCA
import mdtraj as md

from conf.conf_analysis import xyzt
from database.query import traj2grotop
           

def plot_explained_variance(axs, variances, n_components):
    axs.tick_params(labelsize=14)
    # Plot the variance explained by each principal component
    axs.bar(np.arange(n_components), variances[:n_components], color='blue')
    axs.set_xlabel('PC', fontsize=16)
    axs.set_ylabel('Fractional variance', fontsize=16)
    
    axs.set_xlim(-1, n_components)
    axs.set_ylim(0)
    axs.grid(True) 

def plot_cumulative_variance(axs, variances, n_components):
    axs.tick_params(labelsize=14)
    # Plot the cumulative variance explained by each principal component
    axs.scatter(np.arange(n_components), np.cumsum(variances)[:n_components], color='red', s=8)
    axs.set_xlabel('PC', fontsize=16)
    axs.set_ylabel('Cumulative fractional variance', fontsize=16)

    axs.set_xlim(-1, n_components)
    axs.set_ylim(0,1)
    axs.grid(True)

class pca_analysis:
    def __init__(self, xyz_data, labels: pd.DataFrame):
        self.xyz_data = xyz_data
        self.labels = labels

        # Center the data
        # Not actually necessary for PCA implementation here, but it's a good idea for some other purposes
        # i.e. when centered data is needed and accessed from this class
        self.data_center = np.mean(self.xyz_data, axis=0).reshape(1, self.xyz_data.shape[-1])
        self.pca_input = self.xyz_data - self.data_center

    def transform(self, normalize=False, n_components=None):
        if normalize:
            # Normalize the input matrix by the standard deviation
            pca_input = self.pca_input / np.std(self.pca_input, axis=0)
            # Normalize the input matrix by the max value
            # pca_input = pca_input / np.max([np.fabs(np.min(pca_input, axis=0)), np.max(pca_input, axis=0)], axis=0)
        else:
            pca_input = self.pca_input
        # pca = PCA(n_components=self.pca_input.shape[-1])
        if n_components is None:
            pca = IPCA(n_components=self.pca_input.shape[-1])
        else:
            pca = IPCA(n_components=n_components)

        self.pca_output = pca.fit_transform(pca_input)
        self.weight_matrix = pca.components_
        self.variances = pca.explained_variance_ratio_

    def plot_xyzvar(self, axs):
        axs.plot(np.var(self.xyz_data, axis=0))
        axs.set_ylabel('Variance')

    def plot_2dpc(self, axs, val, by='traj_id', comp1=1, comp2=2):
        # Plot two principal components in a 2D scatter plot
        # comp1 and comp2 are the indices of the principal components to plot (one-indexed)
        # val is the value of the column of self.labels to plot by (e.g. a valid value in the 'traj_id' column)
        # by is the column of self.labels to plot by (e.g. 'traj_id')
        
        # Get the indices of the rows of self.labels that have this value
        which = (self.labels[by] == val).to_numpy()
        # Plot the points with this value
        comp1_pts = self.pca_output[which, comp1-1]
        comp2_pts = self.pca_output[which, comp2-1]
        axs.scatter(comp1_pts, comp2_pts, label=val)
        # # Plot the starting point of the trajectory
        # axs.scatter(comp1_pts[0], comp2_pts[0], marker='x', color='black')
        axs.set_xlabel('Principal Component '+str(comp1))
        axs.set_ylabel('Principal Component '+str(comp2))
        axs.legend()

        axs.set_aspect('equal', adjustable='box', anchor='C')

    def plot_explained_variance(self, axs, n_components):
        plot_cumulative_variance(axs, self.variances, n_components)

    def plot_cumulative_variance(self, axs, n_components):
        plot_cumulative_variance(axs, self.variances, n_components)

    def pc_extrema(self, comp):
        # Get the indices of the points that are the most extreme in the given principal component
        # comp is the index of the principal component (one-indexed)
        # Returns a tuple of the indices of the most positive and most negative points
        max_index = np.argmax(self.pca_output[:,comp-1])
        min_index = np.argmin(self.pca_output[:,comp-1])

        extrema = {}

        traj_id, timestep = self.labels.iloc[max_index][['traj_id', 'timestep']]
        print(f'Max: traj_id={traj_id}, timestep={timestep}')
        extrema['max'] = (traj_id, timestep)

        traj_id, timestep = self.labels.iloc[min_index][['traj_id', 'timestep']]
        print(f'Min: traj_id={traj_id}, timestep={timestep}')
        extrema['min'] = (traj_id, timestep)

        return extrema
    
class prot_pca(xyzt):
    def __init__(self, traj_ids, 
                 atomselect='protein and name CA', atomstride=1, 
                 align_atomselect=None):
        super().__init__(traj_ids)

        # TODO: temporary
        self.load_refs()
        
        # atomselect is the atom selection string for the atoms to use in the PCA
        self.atomselect = atomselect
        self.atomstride = atomstride
        
        # The atom indices might depend on the system specified by grotop
        self.select_aindex = {grotop_id: ref.top.select(self.atomselect)[::atomstride] 
                              for grotop_id, ref in self.refs.items()}
        
        if align_atomselect is not None:
            self.align_index = {grotop_id: ref.top.select(align_atomselect) 
                                for grotop_id, ref in self.refs.items()}

    def pca_init(self, normalize=False, n_components=None, tf=None, 
                 align_xyz=None):
        
        self.open_nc()

        # This thing is huge
        # So we won't keep it as an attribute of the class
        # Get coordinates of the selected atoms
        select_coord_set = []

        if tf is None:
            for t in self.traj_ids:
                # Get the coordinates of the selected atoms
                select_index = self.select_aindex[traj2grotop(t)]

                if align_xyz is not None:
                    align_index = self.align_index[traj2grotop(t)]
                    trajxyz = self.getcoords(t, select_index, df=False, align_aindex=align_index, align_xyz=align_xyz)
                else:
                    trajxyz = self.getcoords(t, select_index, df=False)

                select_coord_set.append(trajxyz)
        else:
            for t in tf['traj_id'].unique():
                if t not in self.traj_ids:
                    print(f'Warining: traj_id={t} not in traj_ids. There might be a problem.')

                # Get the timesteps in tf
                timesteps = tf.query('traj_id == @t')['timestep'].values
                # Get the coordinates of the selected atoms
                select_index = self.select_aindex[traj2grotop(t)]

                if align_xyz is not None:
                    align_index = self.align_index[traj2grotop(t)]
                    trajxyz = self.getcoords(t, select_index, df=False, align_aindex=align_index, align_xyz=align_xyz)
                else:
                    trajxyz = self.getcoords(t, select_index, df=False)

                # Get the coordinates of the selected atoms at the timesteps in tf
                trajxyz = trajxyz[timesteps, :, :]
                select_coord_set.append(trajxyz)

        self.close_nc()

        # Concatenate into one
        select_coord_set = np.vstack(select_coord_set)

        # Temp solution: without checking whether atom numbers are the same
        xyz_data = select_coord_set.reshape(select_coord_set.shape[0], -1)
        
        # Might end up being useful
        self.input_var  = np.var(xyz_data, axis=0)
        # # plt.plot(np.var(input_dat, axis=0))
        # fig, axs = plt.subplots()
        # var_distrib = hist1d(np.var(input_dat, axis=0), range=[0,9], bins=100)
        # var_distrib.plot(axs)
        # fluc_var = var_distrib.plot_edges[np.argmax(var_distrib.dens)-2]
        # fluc_var * input_dat.shape[1] / np.sum(np.var(input_dat, axis=0))

        # Initialize the PCA
        if tf is None:
            # Prepare tf
            tf = pd.DataFrame()
            tf['traj_id'] = np.repeat(self.traj_ids, [self.nframes[t] for t in self.traj_ids])
            tf['timestep'] = np.concatenate([np.arange(self.nframes[t]) for t in self.traj_ids])
            
        self.pca = pca_analysis(xyz_data, tf[['traj_id', 'timestep']])
        self.pca.transform(normalize=normalize, n_components=n_components)

    # A function I made to allow me to align the xyz to parts before analysis
    # Focus more on internal components
    def prealign(self, traj_xyzdf: pd.DataFrame, xyz_ref, select_index, align_index):
        traj_xyz = [df[['x', 'y', 'z']].values() for _, df in traj_xyzdf.groupby('timestep')]
        traj_xyz = np.stack(traj_xyz)

    def get_trajpc(self, traj_id, comp):
        # Get the value of the given principal component for the given traj_id
        # comp is the index of the principal component (one-indexed)

        data_where = (self.pca.labels['traj_id'] == traj_id).values
        return self.pca.pca_output[data_where, comp-1]

    def plot_residue_ssw(self, axs, comp, color_dict, spacing_rlabel=15):
        residue_ssw = np.sum(np.split(self.pca.weight_matrix[comp-1,:]**2, 
                                      int(len(self.pca.weight_matrix[comp-1,:])/3)), 
                             axis=1)
        
        axs.bar(np.arange(len(residue_ssw)), residue_ssw, 0.8, color=color_dict)
        axs.set_xlabel('Residue Number')
        axs.set_ylabel('SSW of Residue xyz')
        axs.set_xlim(0, len(residue_ssw))
        axs.set_ylim(0)
        axs.set_xticks(np.arange(0, int(len(residue_ssw)/3), spacing_rlabel))
        # axs.set_xticklabels(residue_list[::spacing_rlabel], rotation='vertical')
        axs.set_title(f'Principal Component {comp}')

    def write_extrema(self, comp, porcupine=True, scaling=2):
        # Write the structures at the extrema of the given principal component
        # comp is the index of the principal component (one-indexed)
        extrema = self.pca.pc_extrema(comp)

        # Handling the min
        traj_id, timestep = extrema['min']
        
        traj_object = self.refs[traj2grotop(traj_id)]
        top2write = traj_object.top.subset(traj_object.top.select("protein"))
        # n_atoms from the traj: A bit dangerous for now
        xyz2write = self.prot_xyz.variables["coordinate"][traj_id, timestep, :top2write.n_atoms, :]

        if porcupine:
            # Prepare coordinates to draw the porcupine plot from
            index2drawfrom = self.select_aindex[traj2grotop(traj_id)]
            xyz2drawfrom = self.prot_xyz.variables["coordinate"][traj_id, timestep, index2drawfrom, :]
            vectors2draw = self.pca.weight_matrix[comp-1,:].reshape(int(len(self.pca.weight_matrix[comp-1,:])/3), 3)
            xyz2drawto = xyz2drawfrom + vectors2draw * scaling

            # Write the porcupine plot drawing script for VMD
            with open(f'porcupine{comp}.tcl', 'w') as file:
                for begin, end in zip(xyz2drawfrom*10, xyz2drawto*10):
                    file.write(f"graphics top cone {{{' '.join(map(str, begin))}}} {{{' '.join(map(str, end))}}} radius [expr 0.2 * 1.7] resolution 10\n")

        # Save min as gro
        extrconfig = md.Trajectory(xyz2write, top2write)
        extrconfig.save_gro(f'comp{comp}.gro')
            
        # Handling the max
        traj_id, timestep = extrema['max']

        traj_object = self.refs[traj2grotop(traj_id)]
        top2write = traj_object.top.subset(traj_object.top.select("protein"))
        # n_atoms from the traj: A bit dangerous for now
        xyz2write = self.prot_xyz.variables["coordinate"][traj_id, timestep, :top2write.n_atoms, :]

        # Save max as xtc
        extrconfig = md.Trajectory(xyz2write, top2write)
        extrconfig.save_xtc(f'comp{comp}.xtc')

        print('vmd -e morph_pca.tcl -args morph_pca.tcl %d' % comp)
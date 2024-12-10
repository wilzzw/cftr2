import os
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import mdtraj as md

from conf.conf_analysis import protca, dist
from plot.plot_utilities import hist2d, hist1d
from utils.core_utilities import always_array, R_x, R_y, R_z
from utils.dataset import read_trajdata, write_trajdata, create_dataset

outer_leaflet_defs = {
    1: [106],
    2: [121],
    3: [219],
    4: [219],
    5: [324],
    6: [334],
    7: [882],
    8: [914],
    9: [1011],
    10: [1011],
    11: [1118],
    12: [1131]
}

inner_leaflet_defs = {
    1: [82],
    2: [142],
    3: [197],
    4: [241],
    5: [303],
    6: [356],
    7: [863],
    8: [938],
    9: [990],
    10: [1033],
    11: [1097],
    12: [1152]
}

class helix_positions(protca):
    def __init__(self, traj_ids, helixnums, leveldefs: dict, **align_kwargs):

        self.helixnums = always_array(helixnums)
        # Collections of residues to represent the helices
        self.leveldefs = {h: leveldefs[h] for h in self.helixnums}
        self.reslevel = np.concatenate(list(self.leveldefs.values()))

        super().__init__(traj_ids, self.reslevel, close_after_init=False)

        # Collect alpha carbon xyz coordinates for the helical position reporters
        self.load_cainfo(**align_kwargs)

        self.position_freq = {}

        # Dataframe to collect COM helix positions
        self.helix_com = pd.DataFrame()

        # traj_id column
        self.helix_com['traj_id'] = np.repeat(self.traj_ids, np.array(list(self.nframes.values()))*len(self.helixnums))
        # timestep column
        self.helix_com['timestep'] = np.repeat(np.concatenate([np.arange(nframes) for nframes in self.nframes.values()]), len(self.helixnums))
        # helix column
        self.helix_com['helix'] = np.tile(self.helixnums, np.sum(list(self.nframes.values())))

        ## Tabulate helix COM positions ##
        for h in self.helixnums:
            # Get the residues representing helix
            helix_resids = self.leveldefs[h]
            # Get COM xyz coordinates of residues representing helix
            ca_xcoords = [self.ca_coord_set.query('resid == @r')['x'].values for r in helix_resids]
            self.helix_com.loc[self.helix_com['helix'] == h, 'x'] = np.mean(ca_xcoords, axis=0)

            ca_ycoords = [self.ca_coord_set.query('resid == @r')['y'].values for r in helix_resids]
            self.helix_com.loc[self.helix_com['helix'] == h, 'y'] = np.mean(ca_ycoords, axis=0)

            ca_zcoords = [self.ca_coord_set.query('resid == @r')['z'].values for r in helix_resids]
            self.helix_com.loc[self.helix_com['helix'] == h, 'z'] = np.mean(ca_zcoords, axis=0)

    def calc_hist2dxy(self, helix, range, bins):
        # Get xy coordinates of the helix
        hcoords = self.helix_com.query('helix == @helix')[['x', 'y']].values
        
        # Calculate 2D histogram
        hist = hist2d(*hcoords.T, range=range, bins=bins)
        self.hist = hist
        return hist
    
    def calc_allhist2dxy(self, range=[[10,35],[40,65]], bins=[100,100]):
        self.range = range
        self.bins = bins
        for h in self.helixnums:
            self.position_freq[h] = self.calc_hist2dxy(h, range, bins)
        return

    def plot_scatter_tspositions(self, axs, traj_id, **plot_kwargs):
        for h in self.helixnums:
            helix_xy = self.helix_com.query('traj_id == @traj_id and helix == @h')[['x', 'y']].values
            axs.scatter(*helix_xy.T, c=np.arange(self.nframes[traj_id]), **plot_kwargs)
        self.xyplot_spec(axs)

    def plot_contour_h(self, axs, helix, cbar_show=False, percentiles=2.**np.arange(-1,7), **plot_kwargs):
        hist = self.position_freq.get(helix, self.calc_hist2dxy(helix, self.range, self.bins))
        plot = hist.dens2d_preset(axs, range=self.range, bins=self.bins, 
                             cbar_show=cbar_show, percentiles=percentiles, **plot_kwargs)
        return plot
    
    def plot_contour(self, axs, indicate_6msm=True, showxy_label=True, cbar_show=True, **kwargs):
        for h in self.helixnums:
            # If h is the last helix, show the colorbar
            if h == self.helixnums[-1]:
                kwargs['cbar_show'] = cbar_show
            self.plot_contour_h(axs, h, **kwargs)
            
        if indicate_6msm:
            # TODO: move to a different separate module to handle these
            pdbstruc = md.load(os.path.expanduser('~/cftr2/data/topologies/6msm_aligned.pdb'))

            for h in self.helixnums:
                helix_resids = self.leveldefs[h]
                cryo_hpos = pdbstruc.xyz[0, pdbstruc.top.select('name CA and resSeq '+' '.join(map(str, helix_resids))), :2]*10
                cryo_hpos = np.mean(cryo_hpos, axis=0)
                axs.scatter([cryo_hpos[0]], [cryo_hpos[1]], marker='x', c='cyan')
        if showxy_label:
            axs.set_xlabel('X [Å]')
            axs.set_ylabel('Y [Å]') 

    def interhelix_dist(self, helix_pair):
        # Make sure h1 is the smaller one
        h1, h2 = np.sort(helix_pair)

        # Get xyz for the two COMs
        h1xyz = self.helix_com.query('helix == @h1')[['x','y','z']]
        h2xyz = self.helix_com.query('helix == @h2')[['x','y','z']]

        # Prepare distance dataframe
        dist_df = pd.DataFrame()
        # Preserve traj_id information
        dist_df['traj_id'] = self.helix_com.iloc[h1xyz.index]['traj_id']
        # Preserve timestep info for ease of query
        dist_df['timestep'] = self.helix_com.iloc[h1xyz.index]['timestep']
        # Register the pair's resid numbers
        dist_df['h1'] = np.repeat(h1, len(h1xyz))
        dist_df['h2'] = np.repeat(h2, len(h2xyz))
        # Register distances
        dist_df['dist'] = dist(h1xyz.to_numpy(), h2xyz.to_numpy())

        return dist_df

    def hdist_level(self):
        self.helixlevel_dists = [self.interhelix_dist(pair) for pair in combinations(self.helixnums, 2)]
        self.helixlevel_dists = pd.concat(self.helixlevel_dists, ignore_index=True)

    def plot_helixlevel_dist(self, axs, traj_id, helix_pair, color='black'):
        nframes = self.nframes.get(traj_id)
        # Make sure h1 is the smaller one
        h1, h2 = np.sort(helix_pair)

        axs.plot(np.arange(nframes), self.helixlevel_dists.query('h1 == @h1 & h2 == @h2 & traj_id == @traj_id')['dist'], c=color)
        axs.set_xlim(0,nframes)
        axs.grid(True, ls='--')

    def plot_alldist_hist(self, axs, helix_pair, hist_range, bins_multiplier=10, **plot_kwargs):
        h1, h2 = np.sort(helix_pair)
        pool_alldist = self.helix_com.query('h1 == @h1 & h2 == @h2')['dist']

        hist = hist1d(pool_alldist, bins=(hist_range[1]-hist_range[0])*bins_multiplier, range=hist_range, density=True)
        hist.plot(axs, **plot_kwargs)


# Helix principal axis
def princ_axis(xyz):
    # Calculate the principal axis of a set of coordinates
    # xyz: (natoms, 3)
    # Return: (3,)

    # Implementation: PCA
    pca = PCA(n_components=3)
    pca.fit(xyz)

    # The principal axis is the first eigenvector
    # Make sure z is positive
    return pca.components_[0]*np.sign(pca.components_[0,2])

class helix_axis(protca):
    def __init__(self, traj_ids, start_resid, end_resid):
        self.start_resid = start_resid
        self.end_resid = end_resid
        super().__init__(traj_ids=traj_ids, resids=np.arange(self.start_resid, self.end_resid+1), close_after_init=False)

        self.data_loc = f"helix_axis/r{self.start_resid}-{self.end_resid}"

    ### I/O functions ###
    def load_data(self, df=True, **read_kwargs):
        # Currently does not supoort stride
        if read_kwargs.get('stride', 1) != 1:
            print("Currently does not support stride")
            return

        axis_data = read_trajdata(self.data_loc, traj_ids=self.traj_ids, **read_kwargs)
        output_dict = {"data": axis_data[0], "nframes": axis_data[1], "traj_ids": axis_data[2]}

        # Append with extra requested attributes
        if len(read_kwargs.get('attrs', [])) > 0:
            output_dict.update({attr: axis_data[3+i] for i, attr in enumerate(read_kwargs['attrs'])})

        if df:
            # Make a dataframe
            output_dict['data'] = pd.DataFrame()
            output_dict['data']['traj_id'] = np.repeat(output_dict['traj_ids'], np.array(output_dict['nframes']))
            output_dict['data']['timestep'] = np.concatenate([np.arange(nframes) for nframes in output_dict['nframes']])
            output_dict['data'][['x', 'y', 'z']] = axis_data[0]

        return output_dict
    
    def write_data(self, **write_kwargs):
        # Write the axis data to hdf5
        for t in self.traj_ids:
            dat = self.princ_axes_df.query("traj_id == @t")[['x', 'y', 'z']].values
            write_trajdata(traj_id=t, datadir=self.data_loc, data=dat, **write_kwargs)

    def init_data(self):
        # This is in case such dataset does not exist
        create_dataset(self.data_loc,
                       shape=(len(self.traj_ids), max(self.nframes.values()), 3),
                       dtype='float32', 
                       traj_ids=self.traj_ids, nframes=np.zeros(len(self.traj_ids), dtype=int))
        return

    # Helper function to calculate the axis from linearized xyz array
    # Help with vectorization
    def _calc_axis(self, xyz1d):
        # Reshape to (natoms, 3)
        xyz = xyz1d.reshape(-1,3)
        # Calculate the principal axis
        return princ_axis(xyz)
    
    def calc_axis(self, traj_id):
        # Calculate the principal axis of the helix over the trajectory
        # traj_id: int
        # Return: (nframes, 3)
        xyz = self.ca_coord_set.query("traj_id == @traj_id")[['x', 'y', 'z']].values

        # Reshape xyz to (nframes, natoms*3)
        xyz1d = xyz.reshape(self.nframes.get(traj_id), -1)
        # Calculate the principal axes
        return np.apply_along_axis(self._calc_axis, 1, xyz1d)
    
    def calc_axis_all(self):
        # Collect CA coordinates for the helical position reporters
        self.load_cainfo()

        xyz = self.ca_coord_set[['x', 'y', 'z']].values
        xyz1d = xyz.reshape(-1, len(self.resids)*3)

        # Calculate the principal axes
        princ_axes = np.apply_along_axis(self._calc_axis, 1, xyz1d)     
        self.princ_axes_df = pd.DataFrame()
        self.princ_axes_df['traj_id'] = np.repeat(self.traj_ids, np.array(list(self.nframes.values())))
        # timestep column
        self.princ_axes_df['timestep'] = np.concatenate([np.arange(nframes) for nframes in self.nframes.values()])
        # axis columns
        self.princ_axes_df[['x', 'y', 'z']] = princ_axes

# Used to calculate the angle between principal axes of two helices, or kink angles
class helix_angle:
    def __init__(self, traj_ids, hrange1, hrange2):
        self.traj_ids = traj_ids
        self.hrange1 = tuple(hrange1)
        self.hrange2 = tuple(hrange2)

    def calc_angle(self, df=True):
        dataset1 = helix_axis(self.traj_ids, *self.hrange1).load_data()['data']
        dataset2 = helix_axis(self.traj_ids, *self.hrange2).load_data()['data']

        assert len(dataset1) == len(dataset2), "The number of frames in the two datasets do not match"

        # Calculate the angle between the principal axes
        # in degrees
        # TODO: a bit problem when product is not in [-1,1]; just over 1 or under -1 due to numerical error
        angle = np.degrees(np.arccos(np.sum(dataset1[['x', 'y', 'z']].values * dataset2[['x', 'y', 'z']].values, axis=1)))

        if df:
            self.angle = dataset1[['traj_id', 'timestep']]
            self.angle['angle'] = angle
            # self.angle['angle'].fillna(0, inplace=True)
        else:
            # Useful for downstream analysis
            self.tf = dataset1[['traj_id', 'timestep']]
            self.angle = angle

def calc_helix_angles(hvec):
    theta = np.degrees(np.arccos(hvec[2]))
    phi = np.degrees(np.arctan2(hvec[1], hvec[0]))
    return theta, phi

    
class kink_analysis:
    def __init__(self, traj_ids, z_seg, w_seg):
        self.traj_ids = traj_ids
        self.z_seg = z_seg
        self.w_seg = w_seg

        # Get the helix axis to be aligned with the z-axis
        z_hvec = helix_axis(self.traj_ids, *self.z_seg)
        self.z_hvec = z_hvec.load_data()['data'][['x', 'y', 'z']]

        # Helix to measure the wobble angle
        w_hvec = helix_axis(self.traj_ids, *self.w_seg)
        self.w_hvec = w_hvec.load_data()['data'][['x', 'y', 'z']]

    def calculate(self, periodic_correction=None):
        self.rotate_system()
        self.theta, self.phi = calc_helix_angles(self.w_hvec_r.T)

        if periodic_correction is not None:
            correction = np.vectorize(periodic_correction)
            self.phi = correction(self.phi)
        return self.theta, self.phi

    def rotate_system(self):
        # Rotation around the z-axis into the yz-plane
        # Keep in radians
        # Need renormalizing to get the angle right
        # range of arccos is always [0, pi]; positive
        phi_z = np.arccos(self.z_hvec['y'] / np.linalg.norm(self.z_hvec[['x', 'y']], axis=-1))
        # Rotation around the x-axis into z-axis
        theta_x = np.arccos(self.z_hvec['z'])

        # Negative sign to get the right rotation when needed (i.e. when the x-component is negative)
        # theta_x does not need a sign change; by construction my z-component of helix axis is positive
        # Refer to right hand rule
        Rmat = R_x(theta_x) @ R_z(phi_z * np.sign(self.z_hvec['x']))
        self.Rmat = Rmat

        # This is the rotated helix axis
        # x and y components should be zero (idea for unit testing)
        z_hvec_r = self.Rmat @ np.expand_dims(self.z_hvec.values, axis=-1)
        self.z_hvec_r = np.squeeze(z_hvec_r, axis=-1)

        # # Sanity check
        # fig, axs = plt.subplots()
        # edgeformat(axs)

        # hist2d(hvec_t[0], hvec_t[1], bins=50).hist2d_contour(axs)
        # axs.set_aspect('equal')
        # axs.set_xlim(-1, 1)
        # axs.set_ylim(-1, 1)



        # Rotated wobbling helix axis
        # self.w_hvec_r = self.Rmat @ self.w_hvec.values.T
        w_hvec_r = self.Rmat @ np.expand_dims(self.w_hvec.values, axis=-1)
        self.w_hvec_r = np.squeeze(w_hvec_r, axis=-1)
        return 

    # Average level alignment
    def rotate_system2(self):
        # Rotation around the z-axis into the yz-plane
        # Keep in radians
        # Need renormalizing to get the angle right
        # range of arccos is always [0, pi]; positive
        phi_z = np.arccos(self.z_hvec['y'].mean() / np.linalg.norm(self.z_hvec[['x', 'y']], axis=-1).mean())
        # Rotation around the x-axis into z-axis
        theta_x = np.arccos(self.z_hvec['z'].mean())

        # Negative sign to get the right rotation when needed (i.e. when the x-component is negative)
        # theta_x does not need a sign change; by construction my z-component of helix axis is positive
        # Refer to right hand rule
        Rmat = R_x(theta_x) @ R_z(phi_z * np.sign(self.z_hvec['x'].mean()))

        # This is the rotated helix axis
        # x and y components should be zero (test idea)
        z_hvec_r = Rmat @ self.z_hvec.values.T
        self.z_hvec_r = np.squeeze(z_hvec_r, axis=0)
        # # Sanity check
        # fig, axs = plt.subplots()
        # edgeformat(axs)

        # hist2d(hvec_t[0], hvec_t[1], bins=50).hist2d_contour(axs)
        # axs.set_aspect('equal')
        # axs.set_xlim(-1, 1)
        # axs.set_ylim(-1, 1)

        self.Rmat = Rmat

        # Rotated wobbling helix axis
        w_hvec_r = self.Rmat @ self.w_hvec.values.T
        self.w_hvec_r = np.squeeze(w_hvec_r, axis=0)
        return 
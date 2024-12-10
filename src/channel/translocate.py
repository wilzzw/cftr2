#from audioop import cross
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mdtraj as md
from netCDF4 import Dataset
import ruptures as rpt

from tslearn.metrics import dtw_path
from scipy.spatial.distance import cdist
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset

from database.query import traj2grotop, get_translocation, transloc_total_time, get_trajattr, project_sql
from plot.plot_utilities import edgeformat
from utils.core_utilities import overlapping_split, consecutive_split
from utils.output_namespace import trajectory, grotop4traj, init_gro
from visual.vmd import beads

ion_atom_selstr = {"CLA": "name CLA or name CL"}

# ion_xyz: source of data from nc file
class ion_xyz_fromnc:
    def __init__(self, traj_ids, ion, readstride=1):
        """ Parameters:
            --- traj_ids: according to the database
            --- ion: the ion type, e.g. 'CLA'
            --- readstride: the stride to read the coordinate data
        """
        self.nc = Dataset(os.path.expanduser(f"~/cftr2/data/xyz/{ion}_xyz.nc"), "r", format="NETCDF4", persist=True)
        # Modifies traj_ids, because not all of them have data in the nc
        traj_ids_avail = self.nc.variables['traj_id'][:]
        traj_ids_avail = traj_ids_avail[~traj_ids_avail.mask]

        # Treat traj_ids
        self.traj_ids = traj_ids
        exclude_traj_ids = np.setdiff1d(self.traj_ids, traj_ids_avail)
        if len(exclude_traj_ids) > 0:
            print("Excluding traj_ids that have no records in NC: {}".format(exclude_traj_ids))
        self.traj_ids_input = self.traj_ids
        self.traj_ids = np.intersect1d(self.traj_ids, traj_ids_avail)

        self.ion = ion
        self.readstride = readstride
        
        stepsize_base = self.nc.variables["time"].units 
        self.timestep = readstride * stepsize_base

        nframes = self.nc.variables['nframes'][self.traj_ids]
        self.nframes = {t: int((f-1) / self.readstride) + 1 for t, f in zip(self.traj_ids, nframes)}

    def _get_aindex(self, traj_id):
        aindex = self.nc.variables['atom_index'][traj_id]
        aindex = aindex[~aindex.mask]
        return aindex

    def _get_coords(self, traj_id):
        # Originally stored number of frames
        nframes = self.nc.variables['nframes'][traj_id]
        # Available aindex
        aindex = self._get_aindex(traj_id)
        # Read coordinates
        coordinates = self.nc.variables['coordinate'][traj_id, :nframes:self.readstride, :len(aindex), :] * 10

        return coordinates, aindex

# ion_xyz: source of data from xtc file
class ion_xyz_fromxtc:
    def __init__(self, traj_ids, ion, readstride=1):
        self.traj_ids = traj_ids

        # Modifies traj_ids, because not all of these trajectories exist
        traj_ids_avail = [t for t in self.traj_ids if os.path.exists(trajectory(t))]
        exclude_traj_ids = np.setdiff1d(self.traj_ids, traj_ids_avail)
        if len(exclude_traj_ids) > 0:
            print("Excluding traj_ids of which the xtc files are not found: {}".format(exclude_traj_ids))
        self.traj_ids_input = self.traj_ids
        self.traj_ids = np.array(traj_ids_avail)

        self.ion = ion
        self.readstride = readstride    
        
    def _get_aindex(self, traj_id):
        aindex = self.refs[traj2grotop(traj_id)].topology.select(ion_atom_selstr[self.ion])
        return aindex
    
    def _get_coords(self, traj_id):
        # Load the trajectory
        traj = md.load(trajectory(traj_id), top=grotop4traj(traj_id), stride=self.readstride)
        nframes = traj.n_frames
        # Get the ion atom indices
        aindex = self._get_aindex(traj_id)
        # Get the coordinates
        coordinates = traj.xyz[:, aindex, :] * 10

        return coordinates, aindex


class ion_xyz(ion_xyz_fromnc, ion_xyz_fromxtc):
    def __init__(self, traj_ids, ion, readstride=1, source='nc'):
        if source == 'nc':
            ion_xyz_fromnc.__init__(self, traj_ids, ion, readstride)
        elif source == 'xtc':
            ion_xyz_fromxtc.__init__(self, traj_ids, ion, readstride)
        else:
            raise ValueError("source must be either 'nc' or 'xtc'")

        self.grotop_ids = sorted(set([traj2grotop(t) for t in self.traj_ids]))
        self.refs = {grotop_id: md.load(init_gro(grotop_id)) for grotop_id in self.grotop_ids}
        
    def get_allcoords(self):
        ion_xyzdf_collect = []
        for t in self.traj_ids:
            ion_xyzdf = pd.DataFrame()

            # Get the coordinates
            coordinates, aindex = self._get_coords(t)
            nframes, nions, _ = coordinates.shape
            # Make df: reshape coordinates to (nframes*len(aindex), 3)        
            # Last dimension should have a length=3 (x, y, z); 3 is less wordy
            coordinates = coordinates.reshape(coordinates.shape[0]*coordinates.shape[1], 3)

            # Insert xyz data into the df
            ion_xyzdf['traj_id'] = np.repeat(t, nframes*nions)
            ion_xyzdf['timestep'] = np.repeat(np.arange(0, nframes), nions)
            ion_xyzdf['aindex'] = np.tile(aindex, nframes)

            ion_xyzdf[['x', 'y', 'z']] = coordinates
            ion_xyzdf_collect.append(ion_xyzdf)
        self.ion_coord_set = pd.concat(ion_xyzdf_collect, ignore_index=True)

### Helper functions ###
# Helper function: produces a boolean array of whether atoms are within a cylinder of specified parameters
def incylinder(atom_coordinates: np.ndarray, xcenter, ycenter, radius, zmin, zmax):
    """
    Returns a boolean array of whether atoms are within a cylinder of specified parameters

    Parameters
    ----------
    atom_coordinates : np.ndarray
        The atom xyz-coordinates array of shape (n_atoms, 3) or (n_timesteps, n_atoms, 3)
    xcenter : float
        The x-coordinate of the center of the cylinder
    ycenter : float
        The y-coordinate of the center of the cylinder
    radius : float
        The radius of the cylinder
    zmin : float
        The z-coordinate of the bottom of the cylinder
    zmax : float
        The z-coordinate of the top of the cylinder

    Returns
    -------
    np.ndarray
        A boolean array of shape (n_timesteps, n_atoms) for atoms within the cylinder
    """
    # xyz part
    assert atom_coordinates.shape[-1] == 3 and atom_coordinates.ndim <= 3

    # Fix the shape if it were less than 3
    if atom_coordinates.ndim != 3:
        atom_coordinates = atom_coordinates.reshape(*np.pad(atom_coordinates.shape, pad_width=(3-atom_coordinates.ndim, 0), constant_values=1))
    
    radii = np.linalg.norm(atom_coordinates[:,:,:2] - np.array([xcenter, ycenter]), axis=2)
    # Radii is less than cutoff? A boolean array for all atoms selected over all timesteps
    within_radii = (radii <= radius)

    # z-Position greater than the bottom of the cylinder? A boolean array for all atoms selected over all timesteps
    in_zbelow = atom_coordinates[:,:,2] >= zmin
    # z-Position less than the top of the cylinder? A boolean array for all atoms selected over all timesteps
    in_zabove = atom_coordinates[:,:,2] <= zmax

    # Meeting all three criteria means the atom is within the cylinder at a given moment
    in_cylinder = np.ma.stack([within_radii, in_zbelow, in_zabove])
    in_cylinder = np.ma.all(in_cylinder, axis=0)
    return in_cylinder

# Helper function for plotting atom positions
# Decide which timesteps to plot for each atom
# Returns an array of shape (n_timesteps, n_atoms) with True for timesteps to be plotted
# Default values used for my own analyses
def timesteps_to_plot(atom_coordinates: np.ndarray, tolerance_window, xcenter=25, ycenter=50, radius=(300**0.5), zmin=90, zmax=150):
    """
    Helper function for plotting atom positions
    Decide which timesteps to plot for each atom
    To facilitate the lineplots of atom positions
    Returns an array of shape (n_timesteps, n_atoms) with True for timesteps to be plotted

    Parameters
    ----------
    atom_coordinates : np.ndarray
        The atom coordinates array of shape (n_atoms, 3) or (n_timesteps, n_atoms, 3)
    tolerance_window : int
        The number of timesteps to include before and after the interval in which the atom is within the cylinder
    xcenter : float
        The x-coordinate of the center of the cylinder
    ycenter : float
        The y-coordinate of the center of the cylinder
    radius : float
        The radius of the cylinder
    zmin : float
        The z-coordinate of the bottom of the cylinder
    zmax : float
        The z-coordinate of the top of the cylinder
    """

    in_cylinder = incylinder(atom_coordinates, xcenter, ycenter, radius, zmin, zmax)
    
    # Ensure masked elements don't mess up ntsteps, natoms
    if np.ma.is_masked(in_cylinder):
        print("timesteps_to_plot(): Warning: masked elements in coordinates detected")
        in_cylinder = in_cylinder[~np.any(in_cylinder.mask, axis=1), :]
        in_cylinder = in_cylinder[:, ~np.any(in_cylinder.mask, axis=0)]

    ntsteps, natoms = in_cylinder.shape

    for i in range(natoms):
        # For the i-th atom
        tsteps_in_cylinder = np.argwhere(in_cylinder[:,i]).flatten()

        # If the default tolerance window is 1
        # In this way, for each timestep t where ion i is inside the cylinder, ensure t-1 and t+1 are also selected and to be plotted
        tsteps_to_plot = np.concatenate([tsteps_in_cylinder + tolerance for tolerance in range(-tolerance_window, tolerance_window+1)])
        # Get rid of overlapping timesteps from the calculation in the above step
        # Retain only unique frames/timesteps
        tsteps_to_plot = np.unique(tsteps_to_plot)

        # Make sure that the resulting timesteps don't overflow the range of timesteps available
        tsteps_to_plot = tsteps_to_plot[(tsteps_to_plot >= 0) & (tsteps_to_plot < ntsteps)]
        
        # Set all to-be-plotted positions to True
        in_cylinder[tsteps_to_plot, i] = True

    # This returns an array with timesteps to be plotted for the atoms
    return in_cylinder

# Fix large periodic jumps for lineplots
# For input array of size 2 $a, if the incremental jump is greater than some amount, fix periodicity
# Returns fixed array useful for plotting lineplot
# zperiod is the periodic cell dimension
def fix_jumps(a, zmin, zmax, zperiod=160, scale_maxjump=1):
    a = np.array(a)
    # a is a 1D-array of size 2
    output_array = a
    out_of_bound = np.any([a > zmax, a < zmin], axis=0)
    in_bound = ~ out_of_bound
    #print(a, in_bound)
    
    # If both are out of bound, temporary solution:
    # By construction with tolerance window=1, these two should necessarily be on opposite sides
    if np.all(out_of_bound):
        output_array[0] = None
        return output_array

    # TODO: This is a bit contentious I think
    # Failed at traj-349 CL-ion 14
    # Though the less the z-range=zmax-azin incorporates bulk aqueous region where ions move really fast, the less likely such jump happens
    max_allowed_jump = scale_maxjump * (zmax - zmin)
    # a is a 1D-array of size 2
    # No jump fixing
    if np.abs(a[1] - a[0]) < max_allowed_jump:
        return a    

    # Fixing based on midpoint location
    # Fixing is just setting the z-value past the jump to an artificial value
    # It is ok if we only show plots between zmin and zmax
    # But it may exaggerate jumps that are not shown in the intended range to plot
    midpoint = (zmin + zmax) / 2
    if a[in_bound] < midpoint:
        output_array[out_of_bound] = output_array[out_of_bound] - zperiod
    else:
        output_array[out_of_bound] = output_array[out_of_bound] + zperiod
    return output_array

# In this function, a is now a 1D-array of any size
# For each increment, if it is greater than some amount, fix periodicity with fix_jumps()
def fix_jumps_long(a, zmin, zmax, zperiod=160, scale_maxjump=1):
    output_array = a
    consecutive_indices = overlapping_split(np.arange(len(a)))
    for indices in consecutive_indices:
        output_array[indices] = fix_jumps(output_array[indices], zmin, zmax, zperiod, scale_maxjump)
    return output_array

# Intermediate helper function that takes in the coordinates and returns the timesteps (intervals) and z-positions (intervals) to be plotted
# Given a set of specified parameters in the arguments
# z-values are fixed for any jumps that would otherwise mess up the lineplot using fix_jumps_long()
def sorted_lineplots(atom_coordinates, tolerance_window=1, xcenter=25, ycenter=50, radius=(300**0.5), zmin=90, zmax=150, scale_maxjump=1):
    to_plot = timesteps_to_plot(atom_coordinates, tolerance_window, xcenter, ycenter, radius, zmin, zmax)
    atoms_zlocation = atom_coordinates[:,:,2]

    all_tintervals = []
    all_zintervals = []

    assert len(to_plot) == len(atoms_zlocation)
    ntsteps, natoms = to_plot.shape
    total_time = ntsteps - 1

    for i in range(natoms):
        # If there's any timestep to be plotted at all
        if np.any(to_plot[:,i]):
            tsteps_to_plot = np.argwhere(to_plot[:,i]).flatten()
            tintervals = consecutive_split(tsteps_to_plot)
            zintervals = [fix_jumps_long(atoms_zlocation[tI, i], zmin, zmax, scale_maxjump=scale_maxjump) for tI in tintervals]

            # Block to address the rare cases where atom is outside the cylinder but within z-range, which mess up with z-plot
            # Introduce artifact just for the appearance of the plot within the (zmin, zmax) range
            for t, zI in enumerate(zintervals):
                # This is rare but happens. The smaller the radius, probably the more likely
                # Another temporary fix: disregard if last timestep of corresponding tI is the last timestep
                # This is not always true if the interval without tolerance of 1 ends at second last timestep 
                # AND that at the last timestep the atom is outside the cylinder but within z-range
                
                # 2022-06-02 also fix for beginning position as well
                # p for position: first or last
                def next2last(p):
                    if p == -1:
                        return p - 1
                    if p == 0:
                        return 1
                    return

                for p in (-1,0):
                    # Next-to position p:
                    pnext = next2last(p)
                    if zI[p] >= zmin and zI[p] <= zmax and tintervals[t][p] != total_time and tintervals[t][p] != 0:
                        # Choices of z-fix: set to zmin or zmax, depending on whichever is closer
                        zfix_to = [zmin, zmax][np.argmin([zI[pnext]-zmin, zmax-zI[pnext]])]
                        zintervals[t][p] = zfix_to
        else:
            tintervals = []
            zintervals = []
    
        all_tintervals.append(tintervals)
        all_zintervals.append(zintervals)

    return all_tintervals, all_zintervals

# Procedure to detect translocation event given timesteps and z-positions arrays (intervals)
# Works with tolerance window of 1
def detect_translocate(tI, zI, cross_pt, low_start, up_start, same_zmarkers=True):
    # going_ec: array of bool; True if Δz > 0 or going towards EC space
    going_ec = np.diff(zI) > 0
    # going_ic: array of bool; True if Δz < 0 or going towards IC space
    going_ic = np.diff(zI) < 0

    # cross2?: array of bool; True if crossing the cross_pt location
    # first, z-values are compared to cross_pt, the point decided to be the boundary between "inside" and "outside"
    # np.diff(zI <> cross_pt) would be greater than 0 when (zI <> cross_pt) goes from False to True
    cross2ec = np.diff(zI > cross_pt) > 0
    cross2ic = np.diff(zI < cross_pt) > 0

    # exit_pt: an array of bool, where True if crossing the cross_pt to EC occurred
    # This is necessary but I forgot why it can't just be cross2?
    # It does have to do with PBC
    exit2ec = going_ec & cross2ec
    exit2ic = going_ic & cross2ic

    # How many instances of exit crossing, towards EC or IC?
    ec_instances = len(np.argwhere(exit2ec).flatten())
    ic_instances = len(np.argwhere(exit2ic).flatten())
    # Not net translocation
    if ec_instances == ic_instances:
        return
    
    # Decide the direction of flux
    if ec_instances > ic_instances:
        direction = 1
    else:
        direction = -1

    # Just choose the boolean array out EC or IC, which ever occurred more (hence the last instance)
    exit_array = [exit2ec, exit2ic][np.argmax([ec_instances, ic_instances])]

    # Gave an exception to the ion trajectories at start initially in (low_start, up_start)
    if (zI[-1] > cross_pt and (zI[0] < low_start or tI[0] == 0)) or (zI[-1] < cross_pt and (zI[0] > up_start or tI[0] == 0)):
        # Want the last one
        i = np.argwhere(exit_array).flatten()[-1]
        t_marker = (tI[i] + tI[i+1]) / 2
        # TODO: This is a temporary fix. Might just deprecate not same_zmarkers
        if same_zmarkers:
            z_marker = cross_pt
        else:
            z_marker = cross_pt
            # z_marker = (zI[i] + zI[i+1]) / 2
        return (t_marker, z_marker, direction)          


class transloc_iontraj(ion_xyz):
    def __init__(self, traj_ids, ion='CLA', stepsize=20):
        """ Parameters:
            --- traj_ids: according to the database
            --- ion: the ion type, e.g. 'CLA'
            --- stepsize: the stepsize of the trajectory in ps
        """
        # TODO: temporary solution: stepsize/20, as I know the stepsize is 20 ps; but it should be read from the dataset
        # TODO: the reason for the related bug is that stepsize is provided to the argument readstride, not stepsize
        super().__init__(traj_ids, ion, int(stepsize/20))

    # Compile a dataframe of ion translocation information
    # Columns: traj_id, ts, atom_index, ion_index, start, end, stepsize (ps), transloc_id
    # TODO: implement direction of the flux?
    def timeseries_transloc_df(self, **cylinder_params):
        timeseries_transloc = []

        for t in self.traj_ids:
            # Skip if no translocation events on record
            translocations = get_translocation(traj_id=t)
            if len(translocations) == 0:
                continue

            # the ion coordinates and indices
            coords, ions = self._get_coords(t)          
            # The nframes should be consistent
            # transloc_total_time() ultimately reads nframes and time stepsize from transloc_history in sql database;
            # which in turn is automatically written by analyze_translocate()
            # if from_nc=True in analyze_translocate(), it is read from ion_xyz()
            
            # Lessen the precision stringency
            assert int((len(coords) - 1) * self.timestep / 1000)*1000 == int(transloc_total_time(t)/1000)*1000 # in ps; some rounding to the left term

            # get the time intervals in which translocations occurred
            ion_in_cylinder = incylinder(coords, **cylinder_params)
            tIs = []
            for i in range(len(ions)):
                # 220824: still added diff back to consecutive_split()
                # tolerance depends on traj time resolution
                # e.g. transloc marked at 825 whose incylinder() might look like: [..., [41250 41251], [41254 41255 41256 41257], ...]
                # diff=1 to grab tIs (not joining with diff>1) is for stepsize = 1 ns, or 1000 ps
                tIs.append(consecutive_split(np.argwhere(ion_in_cylinder[:,i]).flatten(), diff=(1000/self.timestep)))

            # (timestep marked as translocation event, atom index of the ion)
            transloc_onrecord = [(entry['timestep'], entry['stepsize'], entry['aindex'], entry['transloc_id']) for entry in translocations]

            # TODO: tstep and start, end may have different precision at the moment; fix this
            # TODO: columns names are quite a mess
            for tstep, stepsize, ion, transloc_id in transloc_onrecord:
                # It is the i-th ion
                which_ion = np.argwhere(ions == ion).flatten()[0]
                # Get the time intervals in the pore/cylinder for this given ion that underwent translocations
                tIs4theion = tIs[which_ion]
                # Fine tsteps that reflect finer time intervals of the traj
                fine_tstep = int(tstep * stepsize / self.timestep)
                # fine_tstep = tstep
                # Look for & collect the time interval where the marked translocation occurred
                for tI in tIs4theion:
                    start = tI[0]
                    end = tI[-1]
                    if fine_tstep >= start and fine_tstep <= end:
                        timeseries_transloc.append((t, tstep, ion, which_ion, start, end, self.timestep, transloc_id))
                        break
        # Convert to df
        timeseries_transloc = pd.DataFrame(timeseries_transloc, columns=("traj_id", "ts", "atom_index", "ion_index", "start", "end", "timestep", "transloc_id"))

        return timeseries_transloc




# Main class for translocation analysis
class IonTranslocation(ion_xyz):
    def __init__(self, traj_id, ion="CLA", readstride=5):
        super().__init__(traj_id, ion, readstride)
        self.traj_id = self.traj_ids[0]

        self.xyz, self.aindex = self._get_coords(self.traj_id)

    # Open the database and compute the line plots using default parameters
    def prep(self):
        # Connect to database
        self.sqldb = project_sql()
        # Call and produce values directly used for lineplots
        self.plot_tIs, self.plot_zIs = sorted_lineplots(self.xyz)

    # Elementary helper function detecting translocation events for the i-th ion
    # Takes in the index of the ion + the parameters: the z-cross point, low_start, up_start
    # Time and z intervals are calculated by detect_translocate()
    # Returns a list of tuples (timestep, z-cross, aindex)
    def _find_transloc(self, i, cross_pt, low_start, up_start):
        translocs = []
        for tI, zI in zip(self.plot_tIs[i], self.plot_zIs[i]):
            ion_translocs = detect_translocate(tI, zI, cross_pt, low_start, up_start)
            if ion_translocs is not None:
                t_cross, z_cross, direction = ion_translocs
                translocs.append((int(t_cross), int(z_cross), self.aindex[i], direction))
        return translocs
    
    # Find translocation events for all ions
    def find_translocs(self, cross_pt, low_start, up_start):
        translocs = []
        for i in range(len(self.aindex)):
            ion_translocs = self._find_transloc(i, cross_pt, low_start, up_start)
            # ion_translocs is a list of tuples (timestep, z-cross, aindex)
            translocs += ion_translocs
        self.translocs = translocs
        return self.translocs
    
    # Get translocation info from the database
    # This is not calculated from xyz in the instance
    def read_translocs(self, cross_pt):
        translocs = []
        for entry in get_translocation(exclusion=True, traj_id=self.traj_id):
            translocs.append((entry['timestep'], cross_pt, entry['aindex'], entry['direction']))
        self.translocs = translocs
        return self.translocs
    
    # TODO: turn into main()?
    def init_translocs(self, transloc_from_db=True, 
                       cross_pt=130, low_start=120, up_start=140):
        if transloc_from_db:
            translocs = self.read_translocs(cross_pt)
        else:
            translocs = self.find_translocs(cross_pt, low_start, up_start)
        # self.update_db()
        return translocs
    
    # Elementary helper function to plot the translocation traces for the i-th ion
    def _plot_ion_transloc(self, axs, i, cm, mark_transloc=False):
        for tI, zI in zip(self.plot_tIs[i], self.plot_zIs[i]):
            axs.plot(tI, zI, color=cm(1.*(i+1)/len(self.aindex)))

        # Whether to detect translocation events and mark them on the plot
        if mark_transloc:
            # TODO: named list would fit self.translocs structure better
            translocs = [info for info in self.translocs if info[2] == self.aindex[i]]
            for tstep, ztransloc, _, direction in translocs:
                axs.plot(tstep, ztransloc, marker="*", color='black')

    # Preset settings for plotting
    def _plot_preset(self, axs):
        # Plot formatting
        edgeformat(axs, 1, 1)

        axs.set_title("traj_id={}, model={}, E={}, {}f{}".format(*get_trajattr(self.traj_id, 'traj_id', 'model_name', 'voltage', 'extend_from', 'frame_from').iloc[0]))
        axs.set_ylim(90, 150)
        axs.set_xlim(0, 1000*(1000/self.timestep))
        axs.set_xlabel('Time [ns]')
        axs.set_xticks(np.linspace(0, 1000*(1000/self.timestep), 5+1))
        axs.set_xticklabels(np.linspace(0, 1000, 5+1, dtype=int))
        axs.set_yticks(np.arange(90,150+1,10))
        axs.set_yticklabels(np.arange(-40,20+1,10))
        axs.grid(True, ls='--', color='dimgrey')

    def plot_translocs(self, axs, cmap='prism', mark_transloc=False, assign_path=False):
        # Apply preset settings
        self._plot_preset(axs)

        cm = plt.get_cmap(cmap)
        natoms = len(self.aindex)
        # Plot traces for all ions
        for i in range(natoms):
            self._plot_ion_transloc(axs, i, cm)

        if mark_transloc:
            colorfill_dict = {-1: "black", 0: "blue", 1: "magenta", 2: "green"}
            coloredge_dict = {-1: "yellow", 1: "cyan"}
            # self.translocs requires self.find_translocs() or self.read_translocs() to be called first
            for tstep, ztransloc, _, direction in self.translocs:
                # TODO: a little awkward regarding self.translocs
                # self.translocs vs from calculation in the instance may not be consistent and have different info of path_assign
                # Like here: going through all the trouble to get path_assign
                if assign_path:
                    transloc_entry = self.sqldb.fetch_all(table="translocations", 
                                                          traj_id=self.traj_id, timestep=tstep)
                    if len(transloc_entry) == 0:
                        print("Warning: no such translocation on record")
                        print("traj_id=%d; timestep=%d" % (self.traj_id, tstep))
                        path_assign = -1
                    elif len(transloc_entry) > 1:
                        print("Warning: number of translocation on record == %d" % len(transloc_entry))
                        print("traj_id=%d; timestep=%d" % (self.traj_id, tstep))
                        print(transloc_entry[("transloc_id", "path_assign")])
                        path_assign = transloc_entry[0]["path_assign"]
                    else:
                        path_assign = transloc_entry[0]["path_assign"]
                else:
                    path_assign = -1

                axs.plot(tstep, ztransloc, marker="*", markersize=8, markeredgewidth=0.1, markeredgecolor=coloredge_dict.get(direction, "white"), color=colorfill_dict.get(path_assign, "black"))
    
    # In debug, plot the traces for each ion in a separate plot
    def debug(self):
        for i in range(len(self.aindex)):
            _, axs = plt.subplots()
            axs.set_title(i)
            self._plot_ion_transloc(axs, i, cm=plt.get_cmap("prism"), mark_transloc=True)

    # Update the database by adding new translocation events
    def update_db(self):
        history = self.sqldb.fetch_all(table="transloc_history", traj_id=self.traj_id)

        # Get the translocation records from sql db
        transloc_onrecord = [(entry['timestep'], entry['aindex'], entry['direction']) for entry in get_translocation(exclusion=False, traj_id=self.traj_id)]
        # Valid translocation records, exclude the entries with curate code > 1
        valid_transloc_onrecord = [(entry['timestep'], entry['aindex'], entry['direction']) for entry in get_translocation(traj_id=self.traj_id)]

        # Get the translocation records from the present analysis
        # Exclude z-cross value
        ion_translocs = [(timestep, aindex, direction) for timestep, _, aindex, direction in self.translocs]
        unexplained = [transloc for transloc in valid_transloc_onrecord if transloc not in ion_translocs]

        # Any previously recorded translocation not found in the present analysis?
        if len(unexplained) > 0:
            print("Warning: previous record has translocation record not found in the present analysis:\n traj_id=%d; %s \n" % (self.traj_id, unexplained))
            print("You might want to clean up the previous record after this. Confirm by visualizing the translocation traces.")
            # TODO: be nice to print out the transloc_id of the unexplained translocation
        
        # Any new translocation found?
        for transloc in ion_translocs:
            if transloc not in transloc_onrecord:
                tstep, ion, direction = transloc
                print("New translocation found: traj_id=%d, timestep=%d, ion=%s" % (self.traj_id, tstep, ion))
                self.sqldb.insert(table="translocations", 
                                  traj_id=self.traj_id, timestep=tstep, stepsize=self.timestep, aindex=int(ion), direction=direction)

        # Has it been analyzed already before?
        analyzed_before = len(history) > 0
        if not analyzed_before:
            # TODO: nframes
            self.sqldb.insert(table="transloc_history", 
                              traj_id=self.traj_id, 
                              nframes=self.nframes[self.traj_id], 
                              stepsize=self.timestep)
        else:
            self.sqldb.update(table="transloc_history", 
                              update_dict={"nframes": self.nframes[self.traj_id], "stepsize": self.timestep}, 
                              traj_id=self.traj_id)
        self.sqldb.conn.commit()
        return

    # 231116: Update the new column "direction" in the database
    # Temporary function
    # Too much of a hassle to update the update_db() function to reflect the new column
    # e.g. matching everything else but the direction handling
    def update_direction(self):
        for transloc in self.translocs:
            tstep, _, aindex, direction = transloc
            # update
            self.sqldb.update(table="translocations", 
                              update_dict={"direction": direction}, 
                              traj_id=self.traj_id, timestep=tstep, aindex=aindex)
            self.sqldb.update(table="transloc_history", 
                              update_dict={"nframes": self.nframes[self.traj_id], "stepsize": self.timestep}, 
                              traj_id=self.traj_id)
        self.sqldb.conn.commit()
        return

    # TODO: clarify the purpose of this function
    def _aid(self, i, j, k=0):
        _, axs = plt.subplots()
        self._plot_ion_transloc(axs, i, cm=plt.get_cmap("prism"), mark_transloc=False)
        self._plot_preset(axs)

        tI = self.plot_tIs[i][j][k:]
        zI = self.plot_zIs[i][j][k:]

        algo = rpt.Dynp(model="l2").fit(zI)
        result = algo.predict(n_bkps=1)

        rpt.display(np.array(zI), result)

        output_report = {"tI": tI, "zI": zI, "result": result, "axs": axs}
        output_report["t_cross"] = output_report['tI'][output_report['result'][0]]
        output_report["z_cross"] = output_report['zI'][output_report['result'][0]]
        output_report["aindex"] = self.aindex[i]

        items2fetch = ("transloc_id", "timestep", "path_assign")
        query = self.sqldb.fetch_all(table="translocations", items=items2fetch, traj_id=self.traj_id, aindex=self.aindex[i])
        if len(query) > 0:
            query = np.vstack(query)
        query = pd.DataFrame(query, columns=items2fetch)
        output_report["query"] = query
        
        return output_report

# Used to curate the translocation entries in the database
def curate_db(transloc_id, **curate_dict):
    sql = project_sql()
    # Entry to curate
    entry = sql.fetch_all(table="translocations", transloc_id=transloc_id)[0]
    insert_dict = {key: entry[key] for key in entry.keys() if key != "transloc_id"}

    # Insert a new one
    sql.insert(table="translocations", **insert_dict)
    # Update it with the curated values
    # Give curate code of 1
    new_transloc_id = sql.db.lastrowid
    sql.update(table="translocations", 
               update_dict=curate_dict, transloc_id=new_transloc_id)
    sql.update(table="translocations", 
               update_dict={"curated": 1}, transloc_id=new_transloc_id)

    # Update the old one; make curate code of 2 (to be excluded)
    sql.update(table="translocations", 
               update_dict={"curated": 2}, transloc_id=transloc_id)
    sql.conn.commit()


# Does not write to database
def fasttrack_translocate(traj_id, ion="CLA",
                          axs=None, 
                          from_nc=False, 
                          debug=False, 
                          detect_transloc=True, cross_pt=130, low_start=120, up_start=140, 
                          cmap='prism'):
    cm = plt.get_cmap(cmap)
    ion_translocations = []

    # 220825
    color_dict = {-1: "black", 0: "blue", 1: "magenta", 2: "green"}

    # Get coordinates
    # Source coordinates may be directly from trajectory or from nc
    if from_nc:
        stepsize=100
        # 220824 stepsize for analyze_translocate is 1000 ps when default from nc 
        # because that is the default stepsize for trajectories
        ion_data = ion_xyz(traj_ids=traj_id, ion=ion, stepsize=stepsize) 
        ion_index = ion_data._get_aindex(traj_id)
        xyz = ion_data._get_coords(traj_id, df=False)
        ntsteps = ion_data.nframes[traj_id]
    else:
        traj = md.load(trajectory(traj_id), top=grotop4traj(traj_id))
        ion_index = traj.top.select(ion_atom_selstr[ion])
        xyz = traj.xyz[:,ion_index,:] * 10 # gro format in nm: to Ang
        ntsteps = traj.n_frames
        stepsize = traj.timestep

    # Call and produce values directly used for lineplots
    plot_tIs, plot_zIs = sorted_lineplots(xyz)

    natoms = len(ion_index)

    for i in range(natoms):
        # In debug mode, plot the traces for each ion in a separate plot
        if debug:
            _, axs = plt.subplots()
            axs.set_title(i)
        # For each time interval segment and its corresponding z-coord values in the interval
        for tI, zI in zip(plot_tIs[i], plot_zIs[i]):
            if axs is not None:
                axs.plot(tI, zI, color=cm(1.*(i+1)/natoms))

                axs.set_title("traj_id={}, model={}, E={}, {}f{}".format(*get_trajattr(traj_id, 'traj_id', 'model_name', 'voltage', 'extend_from', 'frame_from').iloc[0]))
                axs.set_ylim(90, 150)
                axs.set_xlim(0, 1000*(1000/stepsize))
                axs.set_xlabel('Time [ns]')
                axs.set_xticks(np.linspace(0, 1000*(1000/stepsize), 5+1))
                axs.set_xticklabels(np.linspace(0, 1000, 5+1, dtype=int))
                axs.set_yticks(np.arange(90,150+1,10))
                axs.set_yticklabels(np.arange(-40,20+1,10))
                axs.grid(True, ls='--', color='dimgrey')   

            # Choose to auto-detect translocations
            if detect_transloc:
                translocations = detect_translocate(tI, zI, cross_pt, low_start, up_start)
                if translocations is not None:
                    t_cross, _, _ = translocations
                    # Update: append the time of crossing as well as ion_index of the ion that did
                    ion_translocations.append((int(t_cross), ion_index[i]))

                    if axs is not None:
                        path_assign = -1
                        axs.plot(*translocations, marker="*", markersize=8, markeredgewidth=0.1, markeredgecolor='cyan', color=color_dict.get(path_assign,"black"))

    if detect_transloc:
        # Temporary solution: return ntsteps also
        return ion_translocations, ntsteps, stepsize


### Flux count ###
def flux_count_ts(translocation_tsteps, ntsteps=1000+1):
    ion_counts = np.zeros(ntsteps, dtype=int)
    for t in sorted(translocation_tsteps):
        ion_counts[t:] += 1
    return ion_counts

def flux_count(traj_id, ntsteps):
    # Timestep marked as translocations in this trajectory, from database
    translocation_tsteps = [entry['timestep'] for entry in get_translocation(traj_id)]
    # 220824 deprecated function transloc_ntsteps()
    # # Get number of timesteps of traj from corresponding entry in database
    # ntsteps = transloc_ntsteps(traj_id)
    return flux_count_ts(translocation_tsteps, ntsteps)


### Ion translocation path clustering using DTW ###
# TODO: separate module?
# Helper class for clustering ion translocation paths
class path_cluster:
    def __init__(self):
        self.model = TimeSeriesKMeans(metric='dtw')

    # Train to obtain the parameters of the model
    # Model with the parameters can be saved to hdf5
    def train(self, init_ts, trainX, save=False):
        # Initial centroid time series
        # Use to_time_series_dataset() to compile time series of different time lengths
        if type(init_ts) is str:
            self.init_ts = init_ts
        else:
            self.init_ts = to_time_series_dataset(init_ts)

        self.trainX = to_time_series_dataset(trainX)
        self.model = TimeSeriesKMeans(metric='dtw', init=self.init_ts, n_jobs=40)
        self.km_paths = self.model.fit(self.trainX)
        if save:
            self.km_paths.to_hdf5(save)
        return self.km_paths

    def classify(self, X):
        self.prepX = to_time_series_dataset(X)
        # Predicted results/labels will be stored in self.km_paths.labels_
        return self.km_paths.predict(self.prepX)

    def visualize_paths(self):
        i = 0
        for center in self.km_paths.cluster_centers_:
            # Render some beads
            beads(center, os.path.expanduser("~/cftr2/figures/paths{}.tcl".format(i)))
            i += 1

# Class for clustering ion translocation paths
class ion_path_clusters(transloc_iontraj, path_cluster):
    def __init__(self, traj_ids, ion, pad_tolerance=200, stepsize=20):
        transloc_iontraj.__init__(self, traj_ids, ion, stepsize)
        path_cluster.__init__(self)

        # Frame multiplier
        self.fmultiplier = int(1000 / self.timestep)
        self.pad_tolerance = pad_tolerance

        # Call and store the df for translocations
        self.transloc = self.timeseries_transloc_df(xcenter=25, ycenter=50, radius=(300**0.5), zmin=115, zmax=145)
        # traj_ids with transloc on record
        self.traj_ids_transloc = self.transloc["traj_id"].unique()

    def ion_finecoords(self, traj_id, from_nc=True):
        if from_nc:
            ion_coords, _ = self._get_coords(traj_id)
        else:
            # the ion atom indices
            trajtop = self.refs.get(traj2grotop(traj_id))
            ions = self.ion_index_set.get(traj_id)
            ions = ions[~ions.mask]
            ion_traj = md.load("trajectories/{}_md_fixed_mda-{}.xtc".format(traj_id, self.ion), top=trajtop.atom_slice(ions))
            ion_coords = ion_traj.xyz

        return ion_coords

    def prepare_dataset(self, stride=1):
        # traj_ids with translocation events: grab the ion xyz coordinates from them
        ion_finexyz = {t: self.ion_finecoords(traj_id=t) for t in self.traj_ids_transloc}

        # To collect xyz timeseries
        xyz_set = []
        # For each recorded translocation events
        for i in range(len(self.transloc)):
            transloc_entry = self.transloc.iloc[i]
            # Get traj_id where it is from, ion_index, start frame number & end frame number
            traj_id, ion_index, start, end, = transloc_entry[["traj_id", "ion_index", "start", "end"]]
            # Get the ion coordinate array: $xyz
            xyz = ion_finexyz.get(traj_id)

            # Padding block: pad some extra frames to get good ion trajectories for fitting
            # By good, it means at the end the ion is still within the pore along z (i.e. 115 < z < 145)
            # For DTW to work well, the start and end of the ion trajectory should be comparable in a meaningful way
            for extra in range(1, self.pad_tolerance):
                # Do not exceed maximum number of frames
                if end + extra > 1000*self.fmultiplier:
                    break

                # The constraint block
                zlevel = xyz[end+extra,ion_index,2]
                if zlevel <= 145 and zlevel >= 115:
                    continue
                # else:
                break

            # Collect the info
            ion_pos = xyz[start:end+extra,ion_index,:]
            
            # Start and end zpositions; deciding whether the ion flows outwards or inwards
            startz, endz = xyz[[start,end],ion_index,2]
            # If outflow
            if startz > endz:
                # Time reversal
                ion_pos = ion_pos[::-1]
            ion_pos = ion_pos[::stride]
            xyz_set.append(ion_pos)
        
        self.xyz_set = to_time_series_dataset(xyz_set)
        return self.xyz_set

    def load(self, model_dir=os.path.expanduser("~/cftr2/results/data/channel/path_cluster/model.hdf5")):
        self.km_paths = self.model.from_hdf5(model_dir)
    
    def train(self, custom_init=True, save=False):
        if custom_init:
            super().train(init_ts=self.xyz_set[3:5+1], trainX=self.xyz_set, save=save) ### Can be refined
        else:
            super().train(init_ts="k-means++", trainX=self.xyz_set, save=save)
        return self.km_paths

    def classify(self, X):
        return super().classify(X=X)

    def visualize_paths(self):
        super().visualize_paths()

    def visualize_iontraj(self, i):
        iontraj = self.xyz_set[i][np.all(~np.isnan(self.xyz_set[i]), axis=-1)]
        beads(iontraj, os.path.expanduser("~/cftr2/figures/ion_path.tcl"))
        return

    def check_ion_ztraj(self):
        i = 0
        for xyz in self.xyz_set:
            plt.figure()
            plt.plot(xyz[:,2])
            plt.ylim(90,180)
            plt.title(i)
            i += 1

    def check_path(self, i, j):
        xyz1 = self.xyz_set[i]
        time1 = self.transloc.iloc[0][["start", "end"]]
        xyz2 = self.xyz_set[j]
        time2 = self.transloc.iloc[1][["start", "end"]]

        mat = cdist(xyz1, xyz2)

        path, sim = dtw_path(xyz1, xyz2)

        plt.imshow(mat, origin='lower')
        plt.axis("auto")

        plt.plot([j for _, j in path], [i for i, _ in path], "w-", lw=2.)
        #plt.xticks(np.arange(0, s_y2.shape[0], 5*f), np.arange(t_y2[0], t_y2[1], 5))
        #plt.yticks(np.arange(0, s_y1.shape[0], 5*f), np.arange(t_y1[0], t_y1[1], 5))

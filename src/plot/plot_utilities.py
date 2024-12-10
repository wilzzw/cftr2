import os

import numpy as np
import matplotlib.pyplot as plt

from database.query import get_trajattr

def autolabel_barplots(ax, rectangular_bars):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rectangular_bars:
        height = round(rect.get_height(), 2)
        ax.annotate('{:.2f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height+0.005),
                    xytext=(0,3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', size=16)

def shared_xylabels(fig, xlabel, ylabel, **label_kwargs):
    background = fig.add_subplot(111)
    # Hide everything except the axis labels
    background.spines['top'].set_color('none')
    background.spines['bottom'].set_color('none')
    background.spines['left'].set_color('none')
    background.spines['right'].set_color('none')
    background.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    # Also set the background to completely transparent
    background.patch.set_alpha(0.0)

    if xlabel is not None:
        background.set_xlabel(xlabel, **label_kwargs)
    if ylabel is not None:
        background.set_ylabel(ylabel, **label_kwargs)


def edgeformat(axs, tickwidth=1.2, edgewidth=1.2, pad=8, fontsize=12):
    plt.rcParams.update({'font.family':'Arial'})
    # Support for multiple axes
    axs = np.atleast_1d(axs).flatten()
    for ax in axs:
        ax.tick_params(direction='in', width=tickwidth, pad=pad, labelsize=fontsize)
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(edgewidth)


# TODO: scale, shift, None labels
def xaxis(axs, title, min, max, step, 
           labels=None, scale=None, shift=None,
           **kwargs):
    axs.set_xticks(np.arange(min, max+step, step))
    if labels is not None:
        axs.set_xticklabels(labels, **kwargs)
    elif scale is not None:
        axs.set_xticklabels([f'{scale*label}' for label in axs.get_xticks()], **kwargs)
    elif shift is not None:
        axs.set_xticklabels([f'{label+shift}' for label in axs.get_xticks()], **kwargs)
    elif labels is False:
        axs.set_xticklabels([])
    else:
        axs.set_xticklabels(axs.get_xticks(), **kwargs)
    axs.set_xlabel(title, fontsize=16)

def yaxis(axs, title, min, max, step,
           labels=None, scale=None, shift=None,
           **kwargs):
    axs.set_yticks(np.arange(min, max+step, step))
    if labels is not None:
        axs.set_yticklabels(labels, **kwargs)
    elif scale is not None:
        axs.set_yticklabels([f'{scale*label:.2f}' for label in axs.get_yticks()], **kwargs)
    elif shift is not None:
        axs.set_yticklabels([f'{label+shift:.2f}' for label in axs.get_yticks()], **kwargs)
    elif labels is False:
        axs.set_yticklabels([])
    else:
        axs.set_yticklabels(axs.get_yticks(), **kwargs)
    axs.set_ylabel(title, fontsize=16)

def savefig(filename, figloc=os.path.expanduser('~/cftr2/figures/'), **save_kwargs):
    filename = figloc + filename

    if filename.endswith('pdf'):
        plt.savefig(filename, bbox_inches='tight', **save_kwargs)
    elif filename.endswith('jpg'):
        plt.savefig(filename, dpi=200, **save_kwargs)
    else:
        plt.savefig(filename, **save_kwargs)

def datacomp_pie(traj_ids, nframes_dict, axs=None, *attr_names):
    # Create a dict of simulation lengths (num timesteps)
    tsteps = {}
    for t in traj_ids:
        ntsteps = nframes_dict.get(t, 0)
        if ntsteps > 0:
            ntsteps -= 1
        tsteps[t] = ntsteps
    
    # Get trajectory attributes
    trajattr = get_trajattr(traj_ids)
    # Specifically the attributes of interest
    # Fill null values with 0
    attr_values = trajattr[list(attr_names)].fillna(0)
    # All unique combinations of the attributes: list of tuples of attributes
    attr_value_combs = [tuple(comb) for comb in np.unique(attr_values.to_numpy(), axis=0)]
    # Grouped traj_ids with each combination of attributes
    trajid_grps = {attr: trajattr[ (attr_values == attr).all(axis=1) ]['traj_id'] for attr in attr_value_combs}
    # The amount of total simulation time considered for each combination/group
    amtoftime_grps = {attr: np.sum([tsteps[t] for t in trajid_grps[attr]]) for attr in attr_value_combs}
    
    # Make a pie chart and return the grouped traj_ids and amounts of total simulation time
    if axs is None:
        plt.pie(amtoftime_grps.values(), labels=amtoftime_grps.keys(), autopct='%1.1f%%', startangle=90)
    else:
        axs.pie(amtoftime_grps.values(), labels=amtoftime_grps.keys(), autopct='%1.1f%%', startangle=90)
    return trajid_grps, amtoftime_grps

### Fast track 1d histogram series ###
# easy
class hist1d:
    def __init__(self, x, **hist_kwargs):
        self.x = np.array(x).flatten()
        self.hist_kwargs = hist_kwargs
        self.hist_kwargs['density'] = self.hist_kwargs.get('density', True)

        self.dens, self.edges = np.histogram(x, **hist_kwargs)
        self.plot_edges = (self.edges[1:] + self.edges[:-1]) / 2

        self.densmax = np.max(self.dens)

    def plot(self, axs, **plot_kwargs):
        axs.plot(self.plot_edges, self.dens, **plot_kwargs)

    def plot_v(self, axs, **plot_kwargs):
        axs.plot(self.dens, self.plot_edges, **plot_kwargs)

    def integral_dens(self, a, b):
        # Integral of the density between a and b
        # a and b are the values of the x-axis
        bins_select = np.all([self.plot_edges >= a, self.plot_edges <= b], axis=0)
        return np.sum(self.dens[bins_select]) * np.diff(self.plot_edges)[0]


### Fast track 2d histogram series ###
class hist2d:
    # Only support equal grid spacing and equal sizes of x and y for now
    # Also no need to support non-density histograms
    # Or unequal grid spacing
    def __init__(self, x, y, **hist_kwargs):
        self.x = np.array(x).flatten()
        self.y = np.array(y).flatten()
        self.hist_kwargs = hist_kwargs
        self.hist_kwargs['density'] = self.hist_kwargs.get('density', True)

        self.dens, self.xedges, self.yedges = np.histogram2d(x, y, **hist_kwargs)
        self.xplot_edges = (self.xedges[1:] + self.xedges[:-1]) / 2
        self.yplot_edges = (self.yedges[1:] + self.yedges[:-1]) / 2

        # Indices of the grid cells that each point falls into
        # -1 because the way np.digitize() is written: returning index i such that a[i-1] </= x </= a[i]
        # As a result, np.digitize() always assign to the right
        self.xbin_index = np.digitize(self.x, self.xedges) - 1
        self.ybin_index = np.digitize(self.y, self.yedges) - 1

        self.densmax = np.max(self.dens)

        assert len(self.x) == len(self.y), 'x and y must have the same length'

    def cumulden_lvls(self, percentile_set):
        # Area of each bin; area element
        # unit area of the grid
        dA = np.diff(self.xedges)[0] * np.diff(self.yedges)[0]

        # Minimum and maximum possible density values
        max_grid_value = 1 / dA # This is if all points fall into one grid cell
        total_data_pts = len(self.x)
        min_grid_value = max_grid_value / total_data_pts # Minumum nonzero value possible

        incr = min_grid_value # Give this a special name; all values are multiples of this

        # To bin the density values into sorted density values
        bins = np.arange(min_grid_value - incr/2, max_grid_value+incr, incr)
        # Place the grid density values into bins
        # Return a grid of bin indices
        bin_index = np.digitize(self.dens, bins)
        # This has benefits because all values within each bin would have the same value -- multiples of $min_grid_value
        # Easier vectorized calculation rationale:
        counts_inbins = np.bincount(bin_index.flatten())
        binvals = np.arange(np.max(bin_index)+1) * min_grid_value

        sum_inbins = counts_inbins * binvals * dA
        # Hopefully there are enough bins for this to look continuous enough and accurate
        integral_vs_lvl = np.cumsum(sum_inbins)

        # (Cumulative) percentile values of the contour levels
        # Including 0 and 100
        percentile_values = np.array(percentile_set)
        # Indices of positions where the integral values cross the percentile values
        indices_cutoff = np.searchsorted(integral_vs_lvl, percentile_values/100)
        # integral_vs_lvl[indices_cutoff]
        # The contour levels
        contour_lvls = bins[indices_cutoff]
        self.contour_lvls = contour_lvls

        # What is actually displayed (100% - percentile_values%)
        percentile_display = 100 - percentile_values

        return contour_lvls, percentile_display

    def hist2d_contour(self, axs, **contour_kwargs):
        '''Plot the contour lines of the 2d histogram'''
        '''common contour_kwargs: levels, colors, linestyles, linewidths, alpha'''
        contour = axs.contour(self.xplot_edges, self.yplot_edges, self.dens.T, **contour_kwargs)
        return contour
    
    def hist2d_contourf(self, axs, **contourf_kwargs):
        '''Plot the filled contour of the 2d histogram'''
        '''common contourf_kwargs: levels, colors, alpha, cmap, norm'''
        # Replace zeros with NaNs
        masked_dens = np.ma.masked_where(self.dens == 0, self.dens)
        contourf = axs.contourf(self.xplot_edges, self.yplot_edges, masked_dens.T, **contourf_kwargs)
        return contourf

    # TODO: Contour levels must be increasing is still a problem
    def dens2d_preset(self, axs, no_last_contour_line=True, **kwargs):
        '''Plot the 2d density plot with preset parameters'''
        
        cmap = kwargs.get('cmap', 'hot_r')
        lw = kwargs.get('lw', 1)
        percentiles = kwargs.get('percentiles', np.arange(0, 100+10, 10))
        levels, levelshow = self.cumulden_lvls(percentiles)
        # norm = kwargs.get('norm', mpl.colors.LogNorm())
        # norm = kwargs.get('norm', mpl.colors.Normalize())
        norm = kwargs.get('norm')
        cbar_show = kwargs.get('cbar_show', False)
        
        contour_fill = self.hist2d_contourf(axs, cmap=cmap, levels=levels, norm=norm, extend='min')
        # Zero or one
        level_start = int(no_last_contour_line)
        self.hist2d_contour(axs, levels=levels[level_start:], colors='black', linewidths=lw)
        # hist2d_obj.hist2d_contour(axs, levels=levels, colors='black', linewidths=lw)

        if cbar_show:
            cbar = plt.colorbar(contour_fill, ticks=levels)
            cbar.ax.set_yticklabels(levelshow, size=12)
            cbar.ax.minorticks_off()

        axs.set_aspect('equal', adjustable='box', anchor='C')
        axs.grid(True, linestyle='--')

        edgeformat(axs)

    # TODO: a not so elegant way to handle too many digits in the contour levels if cbar_show is True
    def dens2d_preset2(self, axs, no_last_contour_line=True, level_multiplier_exponent=0, set_aspect=True, **kwargs):
        '''Plot the 2d density plot with preset parameters'''
        
        cmap = kwargs.get('cmap', 'hot_r')
        lw = kwargs.get('lw', 1)
        lmax = kwargs.get('lmax', self.densmax)
        nlevels = kwargs.get('nlevels', 10)
        levels = np.linspace(0, lmax, nlevels)
        norm = kwargs.get('norm')
        cbar_show = kwargs.get('cbar_show', False)
        
        contour_fill = self.hist2d_contourf(axs, cmap=cmap, linewidths=lw, levels=levels, norm=norm, extend='min')
        # Zero or one
        level_start = int(no_last_contour_line)
        self.hist2d_contour(axs, levels=levels[level_start:], colors='black', linewidths=lw)
        # hist2d_obj.hist2d_contour(axs, levels=levels, colors='black', linewidths=lw)

        if set_aspect:
            axs.set_aspect('equal', adjustable='box', anchor='C')
        axs.grid(True, linestyle='--')

        edgeformat(axs)

        if cbar_show:
            cbar = plt.colorbar(contour_fill, ticks=levels)
            # keep up to 1 decimal place
            level_labels = [f'{level*(10**level_multiplier_exponent):.1f}' for level in cbar.get_ticks()]
            cbar.ax.set_yticklabels(level_labels, size=12)
            if level_multiplier_exponent != 0:
                cbar.ax.set_ylabel(r'prob. density [$10^{}$ A.U.]'.format(level_multiplier_exponent), fontsize=16)
            else:
                cbar.ax.set_xlabel('prob. density [A.U.]', fontsize=16)
            cbar.ax.minorticks_off()

            return cbar
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.constants import elementary_charge as e

from utils.dataset import read_trajdata
from database.query import get_trajattr, get_translocation
from plot.plot_utilities import edgeformat, xaxis, yaxis


def iv_fit(to_fit: pd.DataFrame):
    y = to_fit['Iz'].values
    x = to_fit['V'].values
    yerr = to_fit['Iz_err'].values

    def lm(x, b):
        return b*x
    popt, pcov = curve_fit(lm, x, y)
    b = popt
    b_err = np.sqrt(np.diagonal(pcov))
    print(b*1e3*1e12, b_err*1e3*1e12)
    
    return x, lm(x, *popt), b, b_err

class discrete_iv:
    def __init__(self, traj_ids):
        self.traj_ids = traj_ids

    def prepare(self, states_df: pd.DataFrame=None, transloc_path=[]):
        readstride = 50

        # subselect=2 means to get just the z-coordinates (x,y,z): (0,1,2)
        self.boxz, self.nframes, _ = read_trajdata('box', traj_ids=self.traj_ids, subselect=2, stride=readstride)

        self.iv_df = pd.DataFrame()
        self.iv_df['traj_id'] = np.repeat(self.traj_ids, self.nframes)
        self.iv_df['timestep'] = np.hstack([np.arange(n) for n in self.nframes])
        self.iv_df['Ez'] = np.repeat([get_trajattr(t, 'voltage') for t in self.traj_ids], self.nframes)
        self.iv_df['Lz'] = self.boxz

        self.iv_df['V'] = self.iv_df['Ez'] * self.iv_df['Lz'] * 1e3

        self.Ez_vals = sorted(self.iv_df['Ez'].unique())

        #Incorporate states_df information
        if states_df is not None:
            self.iv_df = pd.merge(self.iv_df, states_df, on=['traj_id', 'timestep'])

        ### transloc_df ###
        transloc_df = pd.DataFrame(get_translocation(), columns=get_translocation()[0].keys())
        transloc_df['timestep'] = (transloc_df['timestep'] * transloc_df['stepsize'] / 1000).astype(int)
        transloc_df['stepsize'] = 1000

        if len(transloc_path) > 0:
            transloc_df = transloc_df.query('path_assign in @transloc_path')

        traj_ids = self.traj_ids
        transloc_df = transloc_df.query('traj_id in @traj_ids')

        self.iv_df['index'] = self.iv_df.index
        iv_states_transloc_df = pd.merge(self.iv_df, transloc_df, on=['traj_id', 'timestep'])
        efflux_where = iv_states_transloc_df.query("direction == 1")['index']
        influx_where = iv_states_transloc_df.query("direction == -1")['index']

        self.iv_df['transloc'] = 0
        self.iv_df.loc[efflux_where, 'transloc'] = 1
        self.iv_df.loc[influx_where, 'transloc'] = -1

        # Remove the index column
        self.iv_df.drop(columns='index', inplace=True)

    def compile_iv(self, breakdown_by, breakdown_labels, restraint_string: str=None):
        iv_vals = []
        
        # restraint_string = [f"{state} == {value}" for state, label in restraints.items()]
        # restraint_string = ' & '.join(restraint_string)
        if restraint_string is None:
            restraint_string = 'traj_id >= 0' # Dummy string
        
        for s in breakdown_labels:
            instate_subdf = self.iv_df.query(f"{breakdown_by} == @s & {restraint_string}")

            for Ez in self.Ez_vals:
                subdf = instate_subdf.query("Ez == @Ez")
                avg_V = subdf['V'].mean()
                sem_V = subdf['V'].sem()

                # Total time: in ns
                total_time = len(subdf)

                N_transloc = - subdf['transloc'].sum()
                # Current: 1e9 converts to A
                Iz = (N_transloc * e / total_time) * 1e9
                # Assumes Poisson statistics
                Iz_err = np.abs(Iz / np.sqrt(abs(N_transloc)))

                iv_vals.append([s, Iz, Iz_err, avg_V, sem_V, total_time])

        self.iv_vals = pd.DataFrame(iv_vals, columns=['state_label', 'Iz', 'Iz_err', 'V', 'V_err', 'time'])

    def ivplot(self, axs, state_labels_values: dict, color_dict: dict, states2fit=[]):
        for s, l in state_labels_values.items():
            iv = self.iv_vals.query("state_label == @s")
            axs.scatter(iv['V'], iv['Iz'], s=48, c=color_dict[s], marker="o", edgecolors='black', linewidths=0.2, zorder=3, label=state_labels_values[s])
            # Plot errorbars; fmt='none' prevents points being joined by lines
            # Use the same color as the scatter points
            axs.errorbar(iv['V'], iv['Iz'], yerr=iv['Iz_err'], capsize=6, capthick=1, ecolor=color_dict[s], fmt='none')

            # TODO: fit modularization
            if s in states2fit:
                ## Plot linear fit
                # Negative
                to_fit = iv.query("V <= 0 & V > -120")
                x, yfit, b, b_err = iv_fit(to_fit)
                axs.plot(x, yfit, color=color_dict[s])

                # Positive
                to_fit = iv.query("V >= 0 & V < 120")
                x, yfit, b, b_err = iv_fit(to_fit)
                axs.plot(x, yfit, color=color_dict[s])
    
    def ivplot_preset1(self, axs):
        g_min, g_max = 6, 10 # in pS
        xrange = [-600, 600] # in mV
        yrange = [-8e-12, 8e-12] # in A

        edgeformat(axs, tickwidth=2, edgewidth=2)
        # Vertical zero-axis line
        axs.axvline(0, c='black', zorder=2)
        # Horizontal zero-axis line
        axs.axhline(0, c='black', zorder=2)

        # Experimental
        axs.fill_between(xrange, np.array(xrange)*g_min*1e-3*1e-12, np.array(xrange)*g_max*1e-3*1e-12, color='grey', alpha=0.2)

        xaxis(axs, r'$V$ [mV]', *xrange, 200, fontsize=14)
        yaxis(axs, r'$I$ [pA]', *yrange, 2e-12, scale=1e12, fontsize=14)
        axs.set_xlim(*xrange)
        axs.set_ylim(*yrange)
        axs.set_yticklabels(np.round(np.linspace(*yrange,9)*1e12, 0).astype(int))
        axs.grid(True, ls='--', zorder=3)

    def ivplot_preset2(self, axs):
        g_min, g_max = 6, 10
        xrange = [-120, 120] # in mV
        yrange = [-1.2e-12, 1.2e-12] # in A

        edgeformat(axs, tickwidth=2, edgewidth=2)
        # Vertical zero-axis line
        axs.axvline(0, c='black', zorder=2)
        # Horizontal zero-axis line
        axs.axhline(0, c='black', zorder=2)

        # Experimental
        axs.fill_between(xrange, np.array(xrange)*g_min*1e-3*1e-12, np.array(xrange)*g_max*1e-3*1e-12, color='grey', alpha=0.2)

        xaxis(axs, r'$V$ [mV]', *xrange, 30, fontsize=14)
        yaxis(axs, r'$I$ [pA]', *yrange, 0.3e-12, scale=1e12, fontsize=14)
        axs.set_xlim(*xrange)
        axs.set_ylim(*yrange)
        axs.set_yticklabels(np.round(np.linspace(*yrange,9)*1e12, 1))
        axs.grid(True, ls='--', zorder=3)

    # As an inset
    def ivplot_preset3(self, axs):
        g_min, g_max = 6, 10
        xrange = [-120, 120] # in mV
        yrange = [-1.2e-12, 1.2e-12] # in A

        edgeformat(axs, tickwidth=2, edgewidth=2)
        # Vertical zero-axis line
        axs.axvline(0, c='black', zorder=2)
        # Horizontal zero-axis line
        axs.axhline(0, c='black', zorder=2)

        # Experimental
        axs.fill_between(xrange, np.array(xrange)*g_min*1e-3*1e-12, np.array(xrange)*g_max*1e-3*1e-12, color='grey', alpha=0.2)

        xaxis(axs, r'$V$ [mV]', *xrange, 60, fontsize=20)
        yaxis(axs, r'$I$ [pA]', *yrange, 0.6e-12, scale=1e12, fontsize=20)
        axs.set_xlim(*xrange)
        axs.set_ylim(*yrange)
        axs.set_yticklabels(np.round(np.linspace(*yrange,6)*1e12, 1))
        axs.grid(True, ls='--', zorder=3)


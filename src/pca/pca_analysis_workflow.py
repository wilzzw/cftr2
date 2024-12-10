import matplotlib.pyplot as plt

from pca.pcaIO import load_pca
from clustering.substate_clusters import substates
from plot.plot_utilities import edgeformat, hist1d, hist2d
from visual.vmd import export_snapshots
from visual.visual_utils import pick_state_from_contour

# TODO: should not need traj_ids
class analyze_pca:
    def __init__(self, datadir: str, n_pcs: int, traj_ids):
        self.datadir = datadir
        self.pca_data = load_pca(datadir)
        self.traj_ids = traj_ids

        self.n_pcs = n_pcs
        self.pca_df = self.pca_data.pca_df(self.n_pcs).query('traj_id in @self.traj_ids')

    def variance_plots(self):
        fig, axs = plt.subplots(1, 2, figsize=(10,4), gridspec_kw={'wspace': 0.15})
        self.pca_data.plot_explained_variance(axs[0])
        self.pca_data.plot_cumulative_variance(axs[1])

        axs[0].set_ylim(0, 1)
        return fig, axs
    
    def state_clustering(self, N: int, xrange: list, yrange: list, method: str = 'kmeans', mute_plot=False):
        self.clusters = substates(N, *self.pca_df[['pc1', 'pc2']].values.T)
        
        if method == "kmeans":
            self.clusters.kmeans()
        elif method == "gmm":
            self.clusters.gaussian_mixture()

        if not mute_plot:
            tmpca_clusters = self.clusters.states

            fig, axs = plt.subplots(1, 2, figsize=(10,5), sharey=True, gridspec_kw={'wspace': 0.1})

            # Raw breakdown
            axs[0].scatter(*self.pca_df[['pc1', 'pc2']].values.T, c=tmpca_clusters, s=2)
            axs[0].scatter(*self.clusters.centers.T, marker='x', color='red')

            for i, mu in enumerate(self.clusters.centers):
                axs[0].annotate(i, mu, color='red', fontsize=32)
                
            # Quick contour view
            self.hist = hist2d(*self.pca_df[['pc1', 'pc2']].values.T, bins=60, range=[xrange, yrange])
            self.hist.dens2d_preset2(axs[1], lw=0.5)
            # self.hist.dens2d_preset(axs[1], lw=0.5, percentiles=np.arange(0,100+20,20))

            # Formatting axes
            for ax in axs.flatten():
                edgeformat(ax)
                ax.set_aspect('equal', adjustable='box', anchor='C')
                ax.set_xlim(*xrange)
                ax.set_ylim(*yrange)
                ax.grid(True, linestyle='--')

            return fig, axs
        return
    
    def export_snapshots(self, name: str, s: int, n_sample: int, l: int, visualize: bool = False):
        sampled_snapshots = pick_state_from_contour(self.hist, self.hist.contour_lvls[l], self.clusters.states == s, self.pca_df, n_sample=n_sample, visualize=visualize)
        tf = sampled_snapshots[["traj_id", "timestep"]].values
        export_snapshots(tf, f"{name}.{s}.sh", f"{name}.{s}")

        self.snapshots = sampled_snapshots
        return tf
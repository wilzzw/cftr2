import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as GM
from scipy.spatial.distance import mahalanobis

from utils.analysis_utils import gaussian2d

class substates:
    def __init__(self, nstates, *metrics):
        self.metrics = metrics
        self.data = np.vstack(metrics).T
        self.N = nstates

    def kmeans(self, seed=100, **km_kwargs):
        kmeans = KMeans(n_clusters=self.N, random_state=seed, **km_kwargs).fit(self.data)
        self.km = kmeans
        self.states = kmeans.labels_
        self.centers = kmeans.cluster_centers_
        return
    
    def gaussian_mixture(self, seed=100, **gm_kwargs):
        gmm = GM(n_components=self.N, random_state=seed, **gm_kwargs).fit(self.data)
        self.gmm = gmm
        self.states = gmm.predict(self.data)
        self.centers = gmm.means_
        return

    def indicate_centers(self, axs, xedges, yedges, mdist_lim=1.386):
        i = 0
        for mu, Sigma in zip(self.gmm.means_, self.gmm.covariances_):
            g = gaussian2d(mu, Sigma, xedges, yedges)
            pdf_val_threshold = g.pdf_val_threshold(mdist_lim)
            
            x, y = g.meshgrid
            
            # # Default contour
            # axs.contour(plot_edges, plot_edges, g.G.pdf(np.dstack([x, y])))
            axs.contour(xedges, yedges, g.G.pdf(np.dstack([x, y])), levels=[pdf_val_threshold], linestyles='--', colors=['cyan'])
            
            # Annotation
            axs.annotate(i, mu, color='cyan', fontsize=12)
            i += 1

    def hardgm_states(self, mdist_lim=1.386):
        # Construct an array used to be able to compute the mahalanobis distances to all centers
        # For each snapshot
        # TODO: efficiency evaluation?
        # constructX is of shape (n_samples, n_features, n_components) OR (n_snapshots, n_metrics, n_clusters)
        constructX = np.repeat(self.data.reshape(*self.data.shape,1), self.N, axis=2)

        # Currently only support 2 metrics
        assert self.data.shape[1] == 2

        # Now, pad along the second axis with the index of the cluster
        constructX = np.pad(constructX, pad_width=((0,0),(0,1),(0,0)), constant_values=0)
        for i in range(self.N):
            constructX[:,-1,i] = i
        # constructX done
        # This array is constructed to contain the arguments for the function below (v_mahdist)
            
        # To be vectorized
        def v_mahdist(v):
            d1, d2, i = v
            i = int(i)
            mahdist = mahalanobis(u=self.gmm.means_[i], v=[d1,d2], VI=np.linalg.inv(self.gmm.covariances_[i]))
            return mahdist

        # Returns an array of shape (n_snapshots, n_clusters) containing the mahalanobis distances to all centers
        mahdists2centers = np.apply_along_axis(v_mahdist, 1, constructX)

        hardgm_states = np.full(len(self.data), -1)

        # Assign state to each snapshot: vectorized expression
        within_bool = (mahdists2centers < mdist_lim)

        for grp in range(self.N):
            assign2state = within_bool[:,grp]
            hardgm_states[assign2state] = grp

        return hardgm_states

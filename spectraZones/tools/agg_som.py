# QuickSom modified code for current sklearn version agglomerative clustering

from sklearn.cluster import AgglomerativeClustering
from skimage.feature import peak_local_max
import numpy as np


def agg_cluster_som(som, min_distance=2):
    """
    Perform clustering based on the umatrix.
    Graph-based : using either a minimum spanning tree of the full connectivity

    Parameters
    ----------
    som : SOM object
        The trained SOM object (QuickSom)
    min_distance : int, optional
        Minimum distance between peaks. The default is 2.

    Returns
    -------
    som : SOM object
        The trained SOM object with cluster attribute
    """

    local_min = peak_local_max(-som.umat, min_distance=min_distance)
    n_local_min = local_min.shape[0]
    clusterer = AgglomerativeClustering(
        metric="precomputed", linkage="average", n_clusters=n_local_min
    )
    try:
        labels = clusterer.fit_predict(som.all_to_all_dist)
    except ValueError as e:
        print(
            f'WARNING : The following error was catched : "{e}"\n'
            f"The clusterer yields zero clusters on the data."
            " You should train it more or gather more data"
        )
        labels = np.zeros(som.m * som.n)
    labels = labels.reshape((som.m, som.n))

    som.cluster_att = labels.flatten()
    return som

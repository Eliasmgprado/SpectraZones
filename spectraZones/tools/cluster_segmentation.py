import numpy as np


class ClusterSegmentation:
    """
    Perform Cluster Segmentation by Vote

    """

    def __init__(
        self,
        depths,
        clusters,
        grid_spacing=1,
    ):
        """
        Parameters
        ----------
        depths : np.array
            Array of depths.
        clusters : np.array
            Array of cluster labels.
        grid_spacing : int, optional
            Spacing of new clusters, by default 1 (in depth units).
        """
        self.depths = depths
        self.clusters = clusters
        self.clusters_unique = np.unique(clusters)
        self.grid_spacing = grid_spacing

        self.min_depth = np.min(self.depths)
        self.max_depth = np.max(self.depths)

        self.n_nodes = int((self.max_depth - self.min_depth) // self.grid_spacing)
        self.node_spacing = (self.max_depth - self.min_depth) / self.n_nodes
        self.node_depths = [
            self.min_depth + self.node_spacing * n for n in range(self.n_nodes)
        ] + [self.max_depth]
        self.n_nodes += 1

    def compute_new_clusters(self):

        computed_arr = {}
        node_results = []

        for i, n in enumerate(self.node_depths):

            node_k_count = {cluster: 0 for cluster in self.clusters_unique}
            result = {cluster: {"counts": []} for cluster in self.clusters_unique}

            cell_size = self.node_spacing
            up_limit = n - self.node_spacing

            up_cells = [
                [up_limit + cell_size * j, up_limit + cell_size * (j + 1)]
                for j in range(1)
            ]
            lw_cells = [[n + cell_size * j, n + cell_size * (j + 1)] for j in range(1)]
            cells = up_cells + lw_cells

            total_count = dict(
                zip(self.clusters_unique, np.zeros(len(self.clusters_unique)))
            )

            for cell in cells:
                idxs = np.where((self.depths >= cell[0]) & (self.depths < cell[1]))
                unique, counts = np.unique(self.clusters[idxs], return_counts=True)
                count_ = dict(zip(unique, counts))

                for k in total_count:
                    if k in count_:
                        total_count[k] = max(total_count[k], count_[k])

            for k in result:
                result[k]["counts"].append(total_count[k])

            computed_arr[n] = result

            for k in result:
                x = result[k]["counts"]
                node_k_count[k] = sum(x)

            node_results.append(node_k_count)

        max_count_clusters = list(map(lambda x: max(x, key=x.get), node_results))

        new_clusters = np.zeros(self.clusters.shape)
        for i in range(len(self.node_depths)):

            if i == 0:
                start = self.node_depths[i]
                end = (self.node_depths[i] + self.node_depths[i + 1]) / 2
            elif i == len(self.node_depths) - 1:
                start = (self.node_depths[i - 1] + self.node_depths[i]) / 2
                end = self.node_depths[i]
            else:
                start = (self.node_depths[i - 1] + self.node_depths[i]) / 2
                end = (self.node_depths[i] + self.node_depths[i + 1]) / 2
            idxs = np.where((self.depths >= start) & (self.depths <= end))
            new_clusters[idxs] = str(int(max_count_clusters[i]))

        return new_clusters

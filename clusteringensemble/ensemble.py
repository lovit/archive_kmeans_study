from collections import defaultdict
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans


class KMeansEnsemble:
    def __init__(self, n_ensembles=100, n_clusters=100):
        self.n_ensembles = n_ensembles
        self.n_clusters = n_clusters
    
    def ensemble(self, x, n_final_clusters=10, max_iter=20, verbose=True):
        cooccur = defaultdict(lambda: defaultdict(lambda: 0))
        base_clustering = KMeans(n_clusters=self.n_clusters, n_init=1, max_iter=20)
        for n_iter in range(self.n_ensembles):
            y = base_clustering.fit_predict(x)
            groups = defaultdict(lambda: [])
            for i, yi in enumerate(y):
                groups[yi].append(i)
            for label in groups.keys():
                for i in groups.get(label, []):
                    for j in groups.get(label, []):
                        cooccur[i][j] += 1
        
        rows = []
        cols = []
        data = []
        for i, jdict in cooccur.items():
            for j, v in jdict.items():
                rows.append(i)
                rows.append(j)
                data.append(v)
        
        x_final = csr_matrix(data, (rows, cols))  # TODO: input check
        final_clustering = KMeans(n_clusters=n_final_clusters, 
                                  n_init=1, max_iter=max_iter,
                                  verbose=verbose)
        return final_clustering.fit_predict(x_final)
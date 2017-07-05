from copy import copy
from collections import defaultdict
from collections import namedtuple

Link = namedtuple('Link', 'parent child0 child1 similarity')

from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize


class KMeansEnsemble:
    def __init__(self, n_ensembles=100, n_clusters=100):
        self.n_ensembles = n_ensembles
        self.n_clusters = n_clusters
        self.labels = None
        self.x_final = None
    
    def ensemble(self, x, n_final_clusters=10, max_iter=20, verbose=True, debug_samples=0):
        cooccur = defaultdict(lambda: defaultdict(lambda: 0))
        for n_iter in range(self.n_ensembles):
            base_clustering = KMeans(n_clusters=self.n_clusters, n_init=1, max_iter=20)
            y = base_clustering.fit_predict(x)
            if debug_samples > 0 and n_iter < debug_samples:
                yield y
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
                cols.append(j)
                data.append(v)
        
        x_final = csr_matrix((data, (rows, cols)))
        x_final = normalize(x_final, axis=1, norm='l2')                
        self.x_final = x_final
        
        history, yours = single_linkage(x_final, n_final_clusters)
        labels = {point:label for label, (_, points) in enumerate(yours.items()) for point in points}
        labels = [label for point, label in sorted(labels.items(), key=lambda x:x[0])]
        self.labels = labels
        
        return labels


def single_linkage(similarity_sparse_matrix, n_cluster=2):
    most_similars = []
    
    n = similarity_sparse_matrix.shape[0]
    rows, cols = similarity_sparse_matrix.nonzero()
    data = similarity_sparse_matrix.data
    
    for i, j, d in zip(rows, cols, data):
        if i < j:
            most_similars.append((i, j, d))
    most_similars = sorted(most_similars, key=lambda x:x[2], reverse=True)
    
    i2c = [i for i in range(n)]
    c2set = {i:{i} for i in range(n)}
    new_idx = n
    
    history = []
    yours = None
    
    n_iter = 0
    while len(c2set) > 1 and most_similars:
        # Find a new link
        i, j, sim = most_similars.pop(0)
        ci = i2c[i]
        cj = i2c[j]        
        ij_set = c2set[ci]
        ij_set.update(c2set[cj])
        for p in ij_set:
            i2c[p] = new_idx
        
        c2set[new_idx] = ij_set
        del c2set[ci]
        del c2set[cj]
        
        history.append(Link(new_idx, i, j, sim))
        
        # Remove already merged links
        removal_index = []
        for l, (i, j, _) in enumerate(most_similars):
            if (i in ij_set) and (j in ij_set):
                removal_index.append(l)
        for l in reversed(removal_index):
            del most_similars[l]
        
        # Increase new cluster idx
        new_idx += 1
        
        if len(c2set) == n_cluster:
            yours = {c:copy(ij_set) for c, ij_set in c2set.items()}

        n_iter += 1
        if n_iter > 10000:
            print('too much break')
            break
    
    return history, yours
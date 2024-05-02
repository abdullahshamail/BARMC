from utilities import find_hydrogen_bonds, hb_atom_types_in_frame, extract_semi_vectorized
from barmc_candidate import BARMC_Candidate
import numpy as np
import copy

class BARMCMiner(object):
    """Relaxed Convoy algorithm

    Attributes:
        k (int):  Min number of consecutive timestamps to be considered a convoy
        m (int):  Min number of elements to be considered a convoy
        t1 (int): Max allowed time gap
        t2 (float): Percentage of total time gaps in a relaxed convoy
        threshold (float): Threshold for filtering data
    """
    def __init__(self, clf, k, m, t1, t2, data, threshold, df, atomType, nitrogens_and_oxygens_indices):
        self.clf = clf
        self.k = k
        self.m = m
        self.t1 = t1
        self.t2 = t2
        self.data = data
        self.threshold = threshold
        self.df = df
        self.atomType = atomType
        self.nitrogens_and_oxygens_indices = nitrogens_and_oxygens_indices
        self.HBHashes = {}
        
    def myHashKey(self, t, indices):
        return str(np.concatenate((np.array([t]), np.sort(np.array(indices))), axis=None))

    def createClusters(self, X, y, column, indicesOfFilteredData):
        values = X[:, column, :][indicesOfFilteredData]
        if len(values) < self.m:
            return 1, 0, 0, 0
        clusters = self.clf.fit_predict(values, y=y)
        unique_clusters = set(clusters)
        clusters_indices = {
            cluster: BARMC_Candidate(indices=set(), is_assigned=False, start_time=None, end_time=None, is_cluster_assigned=0)
            for cluster in unique_clusters
        }

        return 0, clusters, unique_clusters, clusters_indices
        
    def checkBAGs(self, atoms, candidate):
        if atoms == []:
            return False
        return (atoms[0] == candidate[0] and atoms[1] == candidate[1] and atoms[2] == candidate[2])
    
    def fit_predict(self, X, y=None):
        BAG_candidates = set()
        columns = len(X[0])
        BAGs = set()
        for column in range(columns):
            results = extract_semi_vectorized(self.data[column], self.nitrogens_and_oxygens_indices, self.threshold)

            reducedMolecules = []
            if len(results) > 0:

                reducedMolecules = np.unique(np.array(results)[:, :2])

            if len(reducedMolecules) > 0:


                indicesOfFilteredData = np.sort(self.df[self.df["subst_id"].isin(reducedMolecules)].index.values)
                tempIndices = np.arange(len(indicesOfFilteredData))
                reverser = dict(zip(tempIndices, indicesOfFilteredData))

            enough_objects, clusters, unique_clusters, clusters_indices = self.createClusters(X, y, column, indicesOfFilteredData)

            if enough_objects == 1 or len(reducedMolecules) == 0:
                continue
            for index, cluster_assignment in enumerate(clusters):
                if cluster_assignment not in clusters_indices:
                    clusters_indices[cluster_assignment] = BARMC_Candidate(indices=set(), is_assigned=False, start_time=None, end_time=None, is_cluster_assigned=0)
                clusters_indices[cluster_assignment].indices.add(reverser[index])


            # update existing convoys
            current_BAG_candidates = set()
            for BAG_cand in BAG_candidates:

                BAG_cand.is_assigned = False
                for cluster in unique_clusters:
                    cluster_indices = clusters_indices[cluster].indices
                    cluster_candidate_intersection = cluster_indices & BAG_cand.indices


                    key = self.myHashKey(column, list(cluster_candidate_intersection))
                    if key in self.HBHashes:
                        HBAtoms = self.HBHashes[key]
                    else:
                        atomsIHave = hb_atom_types_in_frame(cluster_candidate_intersection, self.atomType)
                        HBAtoms = find_hydrogen_bonds(self.data[column], atomsIHave)
                        self.HBHashes[key] = HBAtoms


                    for hb in HBAtoms:
                        present = self.checkBAGs(hb, BAG_cand.atoms)
                        if HBAtoms != [] and len(cluster_candidate_intersection) >= self.m and present:
                            BAG_cand.atoms = hb
                            BAG_cand.indices = cluster_candidate_intersection
                            BAG_cand.end_time = column
                            BAG_cand.gap = 0
                            clusters_indices[cluster].is_cluster_assigned += 1
                            clusters_indices[cluster].atoms = hb
                            BAG_cand.is_assigned = True


                if not BAG_cand.is_assigned:

                    BAG_cand.gap += 1
                    BAG_cand.gaps += 1

                if (BAG_cand.is_assigned) or (BAG_cand.gap <= self.t1):
                    current_BAG_candidates.add(copy.deepcopy(BAG_cand))
                elif BAG_cand.totalLength > self.k and not BAG_cand.is_assigned:
                    BAG_cand.gaps = BAG_cand.gaps - (self.t1 + 1)
                    if(BAG_cand.totalGaps <= self.t2):
                        BAGs.add(copy.deepcopy(BAG_cand))



            # create new candidates
            for cluster in unique_clusters:
                cluster_data = clusters_indices[cluster]
                key = self.myHashKey(column, list(cluster_data.indices))
                if key in self.HBHashes:
                    HBAtoms = self.HBHashes[key]
                else:
                    atomsIHave = hb_atom_types_in_frame(cluster_data.indices, self.atomType)
                    HBAtoms = find_hydrogen_bonds(self.data[column], atomsIHave)
                    self.HBHashes[key] = HBAtoms

                if HBAtoms == []: #if HB not found
                    continue

                if cluster_data.is_cluster_assigned == len(HBAtoms): #cluster is not a new cluster with a new hb
                    continue
                for hb in HBAtoms:
                    if cluster_data.atoms != hb:                    
                        copyCandidate = copy.deepcopy(cluster_data)
                        copyCandidate.start_time = column
                        copyCandidate.end_time = column
                        copyCandidate.gap = 0
                        copyCandidate.gaps = 0
                        copyCandidate.atoms = hb
                        current_BAG_candidates.add(copy.deepcopy(copyCandidate))

            BAG_candidates = current_BAG_candidates
            if column == columns - 1:
                for BAG_cand in current_BAG_candidates:
                    if BAG_cand.totalLength > self.k and BAG_cand.totalGaps <= self.t2:
                        BAGs.add(copy.deepcopy(BAG_cand))
        return BAGs
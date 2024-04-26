import numpy as np
from tqdm import tqdm
from rdkit.ML.Cluster import Butina

def tanimoto_coeff(fp1, fp2):
    """
    This function calculates the tanimoto coefficient between two fingerprints. 
    This is a value between 0 and 1, where 0 is no overlap between molecular substructures and 1 is identical molecules.
    """
    fp1_idx = set(np.flatnonzero(fp1).tolist())
    fp2_idx = set(np.flatnonzero(fp2).tolist())
    return len(fp1_idx.intersection(fp2_idx))/ len(fp1_idx.union(fp2_idx))

def populate_dists_matrix(df):
    """
    Not really a matrix so much as a flat list since thats what rdkit Butina expects
    """
    dists = []
    for i in tqdm(range(len(df.index))):
        fp1 = df.at[i, 'fp']
        for j in range(i):
            fp2 = df.at[j, 'fp']
            dists.append(1 - tanimoto_coeff(fp1, fp2))
    return dists

def get_clusters(mat, num_examples, cutoff):
    return Butina.ClusterData(mat, num_examples, cutoff, isDistData=True)

def get_splits(rand_clusters, all_clusters, num_examples, split = (.8, .1,.1), enforce = None):
    # rand_clusters - shuffled cluster indices
    # all_clusters - dict of counts for each cluster
    # num examples - total # of examples to split, used to determine counts
    # split - 3 elements, fraction of data to use for training, val, test splits respectively
    # enforce - impose a priori certain clusters into certain datasets
    
    cluster_nums = {}
    counts = []
    cluster_idx = 0
    for dataset, split_perc in zip(['train', 'val', 'test'],
                                   split):
        cluster_nums[dataset] = []
        total = 0
        if dataset in enforce:
            #add clusters, update total
            cluster_nums[dataset].append(enforce[dataset]) # fails if more that one, should fix
            total = len(all_clusters[enforce[dataset]])
        while (total < num_examples * split_perc) and (cluster_idx < len(rand_clusters)):
            cluster_nums[dataset] += [rand_clusters[cluster_idx]]
            total += len(all_clusters[rand_clusters[cluster_idx]])
            cluster_idx += 1
        counts.append(total)
    return cluster_nums, counts

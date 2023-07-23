import csv
import numpy as np
import scipy.cluster
import matplotlib.pyplot as plt
import sys

# Function to load the pokemon csv file and return a list of dictionary
def load_data(filepath):
    pokemon_list = [] # result here
    with open(filepath, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            pokemon = {}
            # only extract required key and value
            pokemon['HP'] = row['HP']
            pokemon['Attack'] = row['Attack']
            pokemon['Defense'] = row['Defense']
            pokemon['Sp. Atk'] = row['Sp. Atk']
            pokemon['Sp. Def'] = row['Sp. Def']
            pokemon['Speed'] = row['Speed']
            pokemon_list.append(pokemon)
    return pokemon_list

# turns dict into a feature vector
def calc_features(row):
    features = np.zeros((6,), dtype=np.int64)
    x1 = int(row["Attack"])
    x2 = int(row["Sp. Atk"])
    x3 = int(row["Speed"])
    x4 = int(row["Defense"])
    x5 = int(row["Sp. Def"])
    x6 = int(row["HP"])
    features[0] = x1
    features[1] = x2
    features[2] = x3
    features[3] = x4
    features[4] = x5
    features[5] = x6
    return features

# calculate the Z matrix for clustering
def hac(features):
    clusters = features # reference data locally - treat as all clusters
    Z = np.zeros((len(features)-1, 4)) # result matrix Z
    D = np.zeros((len(features), len(features)), dtype=float) # distance matrix between clusters

    # set D for all current clusters- to be update later after each iteration
    for i in range(len(features)):
        for j in range(len(features)):
            if i != j:
                dist = np.linalg.norm(features[i] - features[j])# stats2 - stats1
                D[i][j] = dist
    
    # all original clusters that already become a cluster - keep track to not cluster again.
    clustered_idx = set()

    # begin clustering here, update 1 row each time.
    for i in range(len(Z)):
        merge_idx1 = -1 # index to be merged, lower one
        merge_idx2 = -1 # 2nd index to be merged
        min_distance = None

        # determine which cluster to merge
        for j in range(len(clusters)):
            # skip already clustered one
            if j in clustered_idx:
                continue

            # get closest distance with current cluster
            distance = D[j]
            min_idx = None
            for k in range(len(distance)):
                if k in clustered_idx:
                    continue
                if k != j and (min_idx == None or distance[k] < distance[min_idx]):
                    min_idx = k
                elif min_idx != None and distance[k] == distance[min_idx] and k != j:
                    #tie breaking
                    min_idx = min(k, min_idx)
            # tie breaking part 2, and check if its smaller than current one
            if min_distance == None or distance[min_idx] < min_distance:
                min_distance = distance[min_idx]
                merge_idx1 = min(j, min_idx)
                merge_idx2 = max(j, min_idx)
            elif distance[min_idx] == min_distance:
                if j < merge_idx1:
                    merge_idx1 = min(j, min_idx)
                    merge_idx2 = max(j, min_idx)
                    min_distance = distance[min_idx]
                elif j == merge_idx1:
                    if min_idx < merge_idx2:
                        merge_idx1 = min(j, min_idx)
                        merge_idx2 = max(j, min_idx)
                        min_distance = distance[min_idx]
            
        # create cluster here, first add to clustered idx, then make cluster by storing as new list
        clustered_idx.add(merge_idx1)
        clustered_idx.add(merge_idx2)
        new_cluster = []
        if type(clusters[merge_idx1]) == list:
            for clus in clusters[merge_idx1]:
                new_cluster.append(clus)
        else:
            new_cluster.append(clusters[merge_idx1])
        if type(clusters[merge_idx2]) == list:
            for clus2 in clusters[merge_idx2]:
                new_cluster.append(clus2)
        else:
            new_cluster.append(clusters[merge_idx2])
        clusters.append(new_cluster)
        
        # update distance of new cluster with other clusters
        with_other = []
        for idx in range(len(clusters)):
            max_dist = None
            # do not update if it is the new one(should be 0) or already in clustered_idx - should be 0
            if idx not in clustered_idx and idx != len(clusters) - 1:
                # case1: both cluster are point
                if type(clusters[idx][0]) == int or type(clusters[idx][0]) == np.int64:
                    if max_dist == None:
                        max_dist = np.linalg.norm(new_cluster[0] - clusters[idx])
                    for new_clust in new_cluster:
                        new_dist = np.linalg.norm(new_clust - clusters[idx])
                        max_dist = max(max_dist, new_dist) # complete linkage
                        #max_dist = min(max_dist, new_dist) # single linkage
                else:
                # case2: cluster is cluster of multiple point
                    if max_dist == None:
                        max_dist = np.linalg.norm(new_cluster[0] - clusters[idx][0])
                    for sub_cluster in clusters[idx]:
                        for new_clust in new_cluster:
                            new_dist = np.linalg.norm(new_clust - sub_cluster)
                            max_dist = max(max_dist, new_dist) # complete linkage
                            #max_dist = min(max_dist, new_dist) # single linkage
            max_dist = 0 if max_dist == None else max_dist # 0 if in cluster or itself
            with_other.append(max_dist)
        with_other = np.array(with_other) # make np array
        
        # using np to stack new distance into the distance matrix
        D = np.vstack((D, np.zeros((1, len(D[0])))))
        D = np.hstack((D, np.zeros((len(D), 1))))
        for n in range(len(with_other)):
            D[len(D) - 1][n] = with_other[n]
            D[n][len(D[0])- 1] = with_other[n]
        
        # update Z matrix
        Z[i][0] = merge_idx1
        Z[i][1] = merge_idx2
        Z[i][2] = min_distance
        Z[i][3] = len(clusters[len(clusters) - 1])
    return Z

# 0.4 to make a dendogram display
def imshow_hac(Z):
    R = scipy.cluster.hierarchy.dendrogram(Z)
    plt.show()

if __name__ == "__main__":
    # get n from command line
    n = int(sys.argv[1])
    # load data
    pokemon_list = load_data("pokemon.csv")
    # calculate Z matrix
    Z = hac([calc_features(row) for row in load_data('Pokemon.csv')][:n])
    # display dendogram
    imshow_hac(Z)